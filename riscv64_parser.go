// Copyright 2022 gorse Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"github.com/klauspost/asmfmt"
	"github.com/samber/lo"
)

// RISCV64Parser implements ArchParser for RISC-V 64-bit architecture
type RISCV64Parser struct{}

// riscv64 regex patterns
var (
	riscv64AttributeLine = regexp.MustCompile(`^\s+\..+$`)
	riscv64NameLine      = regexp.MustCompile(`^\w+:.+$`)
	riscv64LabelLine     = regexp.MustCompile(`^\.\w+_\d+:.*$`)
	riscv64CodeLine      = regexp.MustCompile(`^\s+\w+.+$`)
	riscv64SymbolLine    = regexp.MustCompile(`^\w+\s+<\w+>:$`)
	riscv64DataLine      = regexp.MustCompile(`^\w+:\s+\w+\s+.+$`)

	// Constant pool patterns
	// Match constant pool labels: .LCPI0_0:
	riscv64ConstPoolLabel = regexp.MustCompile(`^\.LCPI\d+_\d+:`)
	// Match .word directive with hex or decimal value
	riscv64WordDirective = regexp.MustCompile(`^\s+\.word\s+(0x[0-9a-fA-F]+|\d+)`)
	// Match .dword directive with hex or decimal value
	riscv64DwordDirective = regexp.MustCompile(`^\s+\.dword\s+(0x[0-9a-fA-F]+|\d+)`)
	// Match section directive for rodata
	riscv64RodataSection = regexp.MustCompile(`^\s+\.section\s+\.rodata`)
	// Match auipc instruction referencing constant pool with %pcrel_hi
	riscv64AuipcConstPool = regexp.MustCompile(`auipc\s+(\w+),\s*%pcrel_hi\(\.LCPI(\d+_\d+)\)`)
	// Match ld/lw instruction with %pcrel_lo constant pool reference
	riscv64LoadConstPoolPcrel = regexp.MustCompile(`(ld|lw|flw|fld)\s+(\w+),\s*%pcrel_lo\(\.LCPI(\d+_\d+)\)\((\w+)\)`)
)

// riscv64 register sets
var (
	riscv64Registers   = []string{"A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"}
	riscv64FPRegisters = []string{"FA0", "FA1", "FA2", "FA3", "FA4", "FA5", "FA6", "FA7"}
)

// riscv64Line represents a single assembly instruction for RISC-V 64-bit
// Binary is string because RISC-V has fixed-width 32-bit instructions
type riscv64Line struct {
	Labels   []string
	Assembly string
	Binary   string
}

// riscv64ConstPool represents a constant pool entry with its label and data
type riscv64ConstPool struct {
	Label string   // e.g., "CPI0_0" (without leading .L)
	Data  []uint32 // Data as 32-bit words (for .word directives)
	Size  int      // Total size in bytes
}

func (line *riscv64Line) String() string {
	var builder strings.Builder
	builder.WriteString("\t")
	if strings.HasPrefix(line.Assembly, "b") {
		splits := strings.Split(line.Assembly, ".")
		op := strings.TrimSpace(splits[0])
		operand := splits[1]
		builder.WriteString(fmt.Sprintf("%s %s", strings.ToUpper(op), operand))
	} else if strings.HasPrefix(line.Assembly, "j") {
		splits := strings.Split(line.Assembly, "\t")
		label := splits[1][1:]
		builder.WriteString(fmt.Sprintf("JMP %s\n", label))
	} else {
		if len(line.Binary) == 8 {
			builder.WriteString(fmt.Sprintf("WORD $0x%v", line.Binary))
		} else {
			_, _ = fmt.Fprintln(os.Stderr, "compressed instructions are not supported.")
			os.Exit(1)
		}
		builder.WriteString("\t// ")
		builder.WriteString(line.Assembly)
	}
	builder.WriteString("\n")
	return builder.String()
}

// Name returns the architecture name
func (p *RISCV64Parser) Name() string {
	return "riscv64"
}

// BuildTags returns the Go build constraint
func (p *RISCV64Parser) BuildTags() string {
	return "//go:build !noasm && riscv64\n"
}

// BuildTarget returns the clang target triple
func (p *RISCV64Parser) BuildTarget(goos string) string {
	return "riscv64-linux-gnu"
}

// CompilerFlags returns architecture-specific compiler flags
func (p *RISCV64Parser) CompilerFlags() []string {
	return []string{"-ffixed-x27"}
}

// Prologue returns C parser prologue (empty for RISC-V - no special types)
func (p *RISCV64Parser) Prologue() string {
	return ""
}

// TranslateAssembly implements the full translation pipeline for RISC-V 64-bit
func (p *RISCV64Parser) TranslateAssembly(t *TranslateUnit, functions []Function) error {
	// Parse assembly
	assembly, stackSizes, constPools, err := p.parseAssembly(t.Assembly)
	if err != nil {
		return err
	}

	// Run objdump
	dump, err := runCommand("objdump", "-d", t.Object)
	if err != nil {
		return err
	}

	// Parse object dump
	if err := p.parseObjectDump(dump, assembly); err != nil {
		return err
	}

	// Copy lines to functions
	for i, fn := range functions {
		if lines, ok := assembly[fn.Name]; ok {
			functions[i].Lines = make([]interface{}, len(lines))
			for j, line := range lines {
				functions[i].Lines[j] = line
			}
		}
		if sz, ok := stackSizes[fn.Name]; ok {
			functions[i].StackSize = sz
		}
	}

	// Generate Go assembly with constant pools
	return p.generateGoAssembly(t, functions, constPools)
}

func (p *RISCV64Parser) parseAssembly(path string) (map[string][]*riscv64Line, map[string]int, map[string]*riscv64ConstPool, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, err
	}
	defer func(file *os.File) {
		if err = file.Close(); err != nil {
			_, _ = fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}(file)

	var (
		stackSizes   = make(map[string]int)
		functions    = make(map[string][]*riscv64Line)
		constPools   = make(map[string]*riscv64ConstPool)
		functionName string
		labelName    string
		// Constant pool parsing state
		inRodataSection   bool
		currentConstPool  *riscv64ConstPool
		currentConstLabel string
	)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()

		// Check for rodata section
		if riscv64RodataSection.MatchString(line) {
			inRodataSection = true
			continue
		}

		// Check for constant pool label (.LCPI0_0:)
		if riscv64ConstPoolLabel.MatchString(line) {
			// Save previous constant pool if any
			if currentConstPool != nil && len(currentConstPool.Data) > 0 {
				constPools[currentConstLabel] = currentConstPool
			}
			// Start new constant pool
			labelPart := strings.Split(line, ":")[0]
			// Normalize label: strip leading .L to get CPI0_0
			labelPart = strings.TrimPrefix(labelPart, ".L")
			currentConstLabel = labelPart
			currentConstPool = &riscv64ConstPool{
				Label: labelPart,
				Data:  make([]uint32, 0),
			}
			continue
		}

		// Parse .word directives for constant pool data
		if inRodataSection || currentConstPool != nil {
			if matches := riscv64WordDirective.FindStringSubmatch(line); matches != nil {
				if currentConstPool != nil {
					val := riscv64ParseIntValue(matches[1])
					currentConstPool.Data = append(currentConstPool.Data, uint32(val))
					currentConstPool.Size += 4
				}
				continue
			}
			if matches := riscv64DwordDirective.FindStringSubmatch(line); matches != nil {
				if currentConstPool != nil {
					val := riscv64ParseIntValue(matches[1])
					// Store dword as two 32-bit words (little-endian)
					currentConstPool.Data = append(currentConstPool.Data, uint32(val), uint32(val>>32))
					currentConstPool.Size += 8
				}
				continue
			}
		}

		// Check for section change that exits rodata section
		if strings.HasPrefix(strings.TrimSpace(line), ".section") && !riscv64RodataSection.MatchString(line) {
			inRodataSection = false
			// Save current constant pool if any
			if currentConstPool != nil && len(currentConstPool.Data) > 0 {
				constPools[currentConstLabel] = currentConstPool
				currentConstPool = nil
				currentConstLabel = ""
			}
		}

		// Check for .text section which also exits rodata
		if strings.HasPrefix(strings.TrimSpace(line), ".text") {
			inRodataSection = false
			if currentConstPool != nil && len(currentConstPool.Data) > 0 {
				constPools[currentConstLabel] = currentConstPool
				currentConstPool = nil
				currentConstLabel = ""
			}
		}

		if riscv64AttributeLine.MatchString(line) {
			continue
		} else if riscv64NameLine.MatchString(line) {
			functionName = strings.Split(line, ":")[0]
			functions[functionName] = make([]*riscv64Line, 0)
			labelName = ""
		} else if riscv64LabelLine.MatchString(line) {
			labelName = strings.Split(line, ":")[0]
			labelName = labelName[1:]
			lines := functions[functionName]
			if len(lines) == 1 || lines[len(lines)-1].Assembly != "" {
				functions[functionName] = append(functions[functionName], &riscv64Line{Labels: []string{labelName}})
			} else {
				lines[len(lines)-1].Labels = append(lines[len(lines)-1].Labels, labelName)
			}
		} else if riscv64CodeLine.MatchString(line) {
			asm := strings.Split(line, "//")[0]
			asm = strings.TrimSpace(asm)
			if labelName == "" {
				functions[functionName] = append(functions[functionName], &riscv64Line{Assembly: asm})
			} else {
				lines := functions[functionName]
				if len(lines) > 0 {
					lines[len(lines)-1].Assembly = asm
				}
				labelName = ""
			}
		}
	}

	// Save any remaining constant pool
	if currentConstPool != nil && len(currentConstPool.Data) > 0 {
		constPools[currentConstLabel] = currentConstPool
	}

	if err = scanner.Err(); err != nil {
		return nil, nil, nil, err
	}
	return functions, stackSizes, constPools, nil
}

// riscv64ParseIntValue parses a decimal or hex integer value from a string
func riscv64ParseIntValue(s string) uint64 {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "0x") || strings.HasPrefix(s, "0X") {
		val, _ := strconv.ParseUint(s[2:], 16, 64)
		return val
	}
	val, _ := strconv.ParseUint(s, 10, 64)
	return val
}

// riscv64GoRegisterName converts RISC-V register names to Go assembly register names
func riscv64GoRegisterName(rvReg string) string {
	rvReg = strings.ToLower(rvReg)
	// Map ABI names to Go register names
	switch rvReg {
	case "a0", "x10":
		return "A0"
	case "a1", "x11":
		return "A1"
	case "a2", "x12":
		return "A2"
	case "a3", "x13":
		return "A3"
	case "a4", "x14":
		return "A4"
	case "a5", "x15":
		return "A5"
	case "a6", "x16":
		return "A6"
	case "a7", "x17":
		return "A7"
	case "t0", "x5":
		return "T0"
	case "t1", "x6":
		return "T1"
	case "t2", "x7":
		return "T2"
	case "t3", "x28":
		return "T3"
	case "t4", "x29":
		return "T4"
	case "t5", "x30":
		return "T5"
	case "t6", "x31":
		return "T6"
	case "fa0", "f10":
		return "FA0"
	case "fa1", "f11":
		return "FA1"
	case "fa2", "f12":
		return "FA2"
	case "fa3", "f13":
		return "FA3"
	case "fa4", "f14":
		return "FA4"
	case "fa5", "f15":
		return "FA5"
	case "fa6", "f16":
		return "FA6"
	case "fa7", "f17":
		return "FA7"
	case "ft0", "f0":
		return "FT0"
	case "ft1", "f1":
		return "FT1"
	case "ft2", "f2":
		return "FT2"
	case "ft3", "f3":
		return "FT3"
	default:
		// Return as uppercase if not recognized
		return strings.ToUpper(rvReg)
	}
}

func (p *RISCV64Parser) parseObjectDump(dump string, functions map[string][]*riscv64Line) error {
	var (
		functionName string
		lineNumber   int
	)
	for i, line := range strings.Split(dump, "\n") {
		line = strings.TrimSpace(line)
		if riscv64SymbolLine.MatchString(line) {
			functionName = strings.Split(line, "<")[1]
			functionName = strings.Split(functionName, ">")[0]
			lineNumber = 0
		} else if riscv64DataLine.MatchString(line) {
			data := strings.Split(line, ":")[1]
			data = strings.TrimSpace(data)
			splits := strings.Split(data, " ")
			var (
				binary   string
				assembly string
			)
			for i, s := range splits {
				if s == "" || unicode.IsSpace(rune(s[0])) {
					assembly = strings.Join(splits[i:], " ")
					assembly = strings.TrimSpace(assembly)
					break
				}
				binary = s
			}
			if lineNumber >= len(functions[functionName]) {
				return fmt.Errorf("%d: unexpected objectdump line: %s", i, line)
			}
			functions[functionName][lineNumber].Binary = binary
			lineNumber++
		}
	}
	return nil
}

func (p *RISCV64Parser) generateGoAssembly(t *TranslateUnit, functions []Function, constPools map[string]*riscv64ConstPool) error {
	// generate code
	var builder strings.Builder
	builder.WriteString(p.BuildTags())
	t.writeHeader(&builder)

	// Emit DATA/GLOBL directives for constant pools
	if len(constPools) > 0 {
		builder.WriteString("\n#include \"textflag.h\"\n")
		builder.WriteString("\n// Constant pool data\n")
		for label, pool := range constPools {
			// Emit DATA directive with little-endian byte order
			// Format: DATA symbol<>+offset(SB)/size, $value
			for i, val := range pool.Data {
				builder.WriteString(fmt.Sprintf("DATA %s<>+%d(SB)/4, $0x%08x\n", label, i*4, val))
			}
			// Emit GLOBL directive to define the symbol size
			builder.WriteString(fmt.Sprintf("GLOBL %s<>(SB), (RODATA|NOPTR), $%d\n", label, pool.Size))
		}
	}

	for _, function := range functions {
		// Calculate return size based on type
		returnSize := 0
		if function.Type != "void" {
			if sz, ok := supportedTypes[function.Type]; ok {
				returnSize = sz // Use actual scalar type size
			} else {
				returnSize = 8 // Default 8-byte slot for pointers/unknown types
			}
		}
		builder.WriteString(fmt.Sprintf("\nTEXT ·%v(SB), $%d-%d\n",
			function.Name, returnSize, len(function.Parameters)*8))
		registerCount, fpRegisterCount, offset := 0, 0, 0
		var stack []lo.Tuple2[int, Parameter]
		for _, param := range function.Parameters {
			sz := 8
			if param.Pointer {
				sz = 8
			} else {
				sz = supportedTypes[param.Type]
			}
			if offset%sz != 0 {
				offset += sz - offset%sz
			}
			if !param.Pointer && (param.Type == "double" || param.Type == "float") {
				if fpRegisterCount < len(riscv64FPRegisters) {
					if param.Type == "double" {
						builder.WriteString(fmt.Sprintf("\tMOVD %s+%d(FP), %s\n", param.Name, offset, riscv64FPRegisters[fpRegisterCount]))
					} else {
						builder.WriteString(fmt.Sprintf("\tMOVF %s+%d(FP), %s\n", param.Name, offset, riscv64FPRegisters[fpRegisterCount]))
					}
					fpRegisterCount++
				} else {
					stack = append(stack, lo.Tuple2[int, Parameter]{A: offset, B: param})
				}
			} else {
				if registerCount < len(riscv64Registers) {
					if param.Type == "_Bool" {
						builder.WriteString(fmt.Sprintf("\tMOVB %s+%d(FP), %s\n", param.Name, offset, riscv64Registers[registerCount]))
					} else {
						builder.WriteString(fmt.Sprintf("\tMOV %s+%d(FP), %s\n", param.Name, offset, riscv64Registers[registerCount]))
					}
					registerCount++
				} else {
					stack = append(stack, lo.Tuple2[int, Parameter]{A: offset, B: param})
				}
			}
			offset += sz
		}
		if offset%8 != 0 {
			offset += 8 - offset%8
		}
		frameSize := 0
		if len(stack) > 0 {
			for i := 0; i < len(stack); i++ {
				if stack[i].B.Pointer {
					frameSize += 8
				} else {
					frameSize += supportedTypes[stack[i].B.Type]
				}
			}
			builder.WriteString(fmt.Sprintf("\tADDI -%d, SP, SP\n", frameSize))
			stackoffset := 0
			for i := 0; i < len(stack); i++ {
				builder.WriteString(fmt.Sprintf("\tMOV %s+%d(FP), T0\n", stack[i].B.Name, frameSize+stack[i].A))
				builder.WriteString(fmt.Sprintf("\tMOV T0, %d(SP)\n", stackoffset))
				if stack[i].B.Pointer {
					stackoffset += 8
				} else {
					stackoffset += supportedTypes[stack[i].B.Type]
				}
			}
		}

		// First pass: find auipc instructions that set up constant pool addresses
		// and track which register they use
		constPoolRegs := make(map[string]string) // baseReg -> constLabel

		for _, lineIface := range function.Lines {
			line, ok := lineIface.(*riscv64Line)
			if !ok {
				continue
			}
			if matches := riscv64AuipcConstPool.FindStringSubmatch(line.Assembly); matches != nil {
				baseReg := strings.ToLower(matches[1])
				constLabel := "CPI" + matches[2]
				if _, hasPool := constPools[constLabel]; hasPool {
					constPoolRegs[baseReg] = constLabel
				}
			}
		}

		// Convert interface{} lines back to riscv64Line
		for _, lineIface := range function.Lines {
			line, ok := lineIface.(*riscv64Line)
			if !ok {
				continue
			}

			// Skip auipc instructions that reference constant pools (they're no longer needed)
			if matches := riscv64AuipcConstPool.FindStringSubmatch(line.Assembly); matches != nil {
				constLabel := "CPI" + matches[2]
				if _, hasPool := constPools[constLabel]; hasPool {
					// Emit any labels that were on this line
					for _, label := range line.Labels {
						builder.WriteString(label)
						builder.WriteString(":\n")
					}
					continue
				}
			}

			// Replace ld/lw/flw/fld instructions that load from constant pools
			if matches := riscv64LoadConstPoolPcrel.FindStringSubmatch(line.Assembly); matches != nil {
				loadOp := matches[1]    // ld, lw, flw, fld
				destReg := matches[2]   // destination register
				constLabel := "CPI" + matches[3]
				baseReg := strings.ToLower(matches[4])
				if _, hasPool := constPools[constLabel]; hasPool {
					// Emit any labels
					for _, label := range line.Labels {
						builder.WriteString(label)
						builder.WriteString(":\n")
					}
					// Emit load address of constant pool into the base register
					builder.WriteString(fmt.Sprintf("\tMOV $%s<>(SB), %s\n",
						constLabel, riscv64GoRegisterName(baseReg)))
					// Emit load from the address based on operation type
					switch loadOp {
					case "ld":
						// 64-bit integer load
						builder.WriteString(fmt.Sprintf("\tMOV (%s), %s\n",
							riscv64GoRegisterName(baseReg), riscv64GoRegisterName(destReg)))
					case "lw":
						// 32-bit integer load (sign-extended)
						builder.WriteString(fmt.Sprintf("\tMOVW (%s), %s\n",
							riscv64GoRegisterName(baseReg), riscv64GoRegisterName(destReg)))
					case "flw":
						// 32-bit float load
						builder.WriteString(fmt.Sprintf("\tMOVF (%s), %s\n",
							riscv64GoRegisterName(baseReg), riscv64GoRegisterName(destReg)))
					case "fld":
						// 64-bit double load
						builder.WriteString(fmt.Sprintf("\tMOVD (%s), %s\n",
							riscv64GoRegisterName(baseReg), riscv64GoRegisterName(destReg)))
					}
					continue
				}
			}

			for _, label := range line.Labels {
				builder.WriteString(label)
				builder.WriteString(":\n")
			}
			if line.Assembly == "ret" {
				if frameSize > 0 {
					builder.WriteString(fmt.Sprintf("\tADDI %d, SP, SP\n", frameSize))
				}
				if function.Type != "void" {
					switch function.Type {
					case "int64_t", "long":
						builder.WriteString(fmt.Sprintf("\tMOV A0, result+%d(FP)\n", offset))
					case "_Bool":
						builder.WriteString(fmt.Sprintf("\tMOVB A0, result+%d(FP)\n", offset))
					case "double":
						builder.WriteString(fmt.Sprintf("\tMOVD FA0, result+%d(FP)\n", offset))
					case "float":
						builder.WriteString(fmt.Sprintf("\tMOVF FA0, result+%d(FP)\n", offset))
					default:
						return fmt.Errorf("unsupported return type: %v", function.Type)
					}
				}
				builder.WriteString("\tRET\n")
			} else {
				builder.WriteString(line.String())
			}
		}
	}

	// write file
	f, err := os.Create(t.GoAssembly)
	if err != nil {
		return err
	}
	defer func(f *os.File) {
		if err = f.Close(); err != nil {
			_, _ = fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}(f)
	bytes, err := asmfmt.Format(strings.NewReader(builder.String()))
	if err != nil {
		return err
	}
	_, err = f.Write(bytes)
	return err
}

func init() {
	RegisterParser("riscv64", &RISCV64Parser{})
}
