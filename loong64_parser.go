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

// Loong64Parser implements ArchParser for LoongArch64 architecture
type Loong64Parser struct{}

// loong64 regex patterns
var (
	loong64AttributeLine = regexp.MustCompile(`^\s+\..+$`)
	loong64NameLine      = regexp.MustCompile(`^\w+:.+$`)
	loong64LabelLine     = regexp.MustCompile(`^\.\w+_\d+:.*$`)
	loong64CodeLine      = regexp.MustCompile(`^\s+\w+.+$`)
	loong64SymbolLine    = regexp.MustCompile(`^\w+\s+<\w+>:$`)
	loong64DataLine      = regexp.MustCompile(`^\w+:\s+\w+\s+.+$`)

	// Constant pool patterns
	// Match constant pool labels: .LCPI0_0:
	loong64ConstPoolLabel = regexp.MustCompile(`^\.LCPI\d+_\d+:`)
	// Match .word directive with hex or decimal value (32-bit)
	loong64WordDirective = regexp.MustCompile(`^\s+\.word\s+(\d+|0x[0-9a-fA-F]+)`)
	// Match .dword directive with hex or decimal value (64-bit)
	loong64DwordDirective = regexp.MustCompile(`^\s+\.dword\s+(\d+|0x[0-9a-fA-F]+)`)
	// Match pcaddu12i instruction referencing constant pool: pcaddu12i $r4, %pc_hi20(.LCPI0_0)
	loong64PcadduConstPool = regexp.MustCompile(`pcaddu12i\s+(\$\w+),\s*%pc_hi20\(\.LCPI(\d+_\d+)\)`)
	// Match ld.w/ld.d instruction with constant pool reference: ld.d $r4, $r4, %pc_lo12(.LCPI0_0)
	loong64LdConstPool = regexp.MustCompile(`ld\.[wd]\s+(\$\w+),\s*(\$\w+),\s*%pc_lo12\(\.LCPI(\d+_\d+)\)`)
)

// loong64 register sets
var (
	loong64Registers   = []string{"R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"}
	loong64FPRegisters = []string{"F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"}
)

// loong64 register aliases (ABI names to Go names)
var loong64RegistersAlias = map[string]string{
	"$zero": "R0",
	"$ra":   "R1",
	"$tp":   "R2",
	"$sp":   "R3",
	"$a0":   "R4",
	"$a1":   "R5",
	"$a2":   "R6",
	"$a3":   "R7",
	"$a4":   "R8",
	"$a5":   "R9",
	"$a6":   "R10",
	"$a7":   "R11",
	"$t0":   "R12",
	"$t1":   "R13",
	"$t2":   "R14",
	"$t3":   "R15",
	"$t4":   "R16",
	"$t5":   "R17",
	"$t6":   "R18",
	"$t7":   "R19",
	"$t8":   "R20",
	"$fp":   "R22",
	"$s0":   "R23",
	"$s1":   "R24",
	"$s2":   "R25",
	"$s3":   "R26",
	"$s4":   "R27",
	"$s5":   "R28",
	"$s6":   "R29",
	"$s7":   "R30",
	"$s8":   "R31",
	"$s9":   "R22",
}

// loong64 operation aliases
var loong64OpAlias = map[string]string{
	"b":    "JMP",
	"bnez": "BNE",
}

// loong64Line represents a single assembly instruction for LoongArch64
// Binary is string because LoongArch has fixed-width 32-bit instructions
type loong64Line struct {
	Labels   []string
	Assembly string
	Binary   string
}

// loong64ConstPool represents a constant pool entry with its label and data
type loong64ConstPool struct {
	Label string   // e.g., "CPI0_0" (without leading .L)
	Data  []uint32 // Data as 32-bit words (for .word directives)
	Size  int      // Total size in bytes
}

func (line *loong64Line) String() string {
	var builder strings.Builder
	builder.WriteString("\t")
	if strings.HasPrefix(line.Assembly, "b") && !strings.HasPrefix(line.Assembly, "bstrins") {
		splits := strings.Split(line.Assembly, ".")
		op := strings.TrimSpace(splits[0])
		registers := strings.FieldsFunc(op, func(r rune) bool {
			return unicode.IsSpace(r) || r == ','
		})
		if o, ok := loong64OpAlias[registers[0]]; !ok {
			builder.WriteString(strings.ToUpper(registers[0]))
		} else {
			builder.WriteString(o)
		}
		builder.WriteRune(' ')
		for i := 1; i < len(registers); i++ {
			if r, ok := loong64RegistersAlias[registers[i]]; !ok {
				_, _ = fmt.Fprintln(os.Stderr, "unexpected register alias:", registers[i])
				os.Exit(1)
			} else {
				builder.WriteString(r)
				builder.WriteRune(',')
			}
		}
		builder.WriteString(splits[1])
	} else {
		builder.WriteString("\t")
		builder.WriteString(fmt.Sprintf("WORD $0x%v", line.Binary))
		builder.WriteString("\t// ")
		builder.WriteString(line.Assembly)
	}
	builder.WriteString("\n")
	return builder.String()
}

// Name returns the architecture name
func (p *Loong64Parser) Name() string {
	return "loong64"
}

// BuildTags returns the Go build constraint
func (p *Loong64Parser) BuildTags() string {
	return "//go:build !noasm && loong64\n"
}

// BuildTarget returns the clang target triple
func (p *Loong64Parser) BuildTarget(goos string) string {
	return "loongarch64-linux-gnu"
}

// CompilerFlags returns architecture-specific compiler flags
func (p *Loong64Parser) CompilerFlags() []string {
	return nil // LoongArch64 doesn't need special fixed-register flags
}

// Prologue returns C parser prologue for architecture-specific types
func (p *Loong64Parser) Prologue() string {
	return "" // LoongArch64 doesn't have special vector types to define
}

// TranslateAssembly implements the full translation pipeline for LoongArch64
func (p *Loong64Parser) TranslateAssembly(t *TranslateUnit, functions []Function) error {
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

func (p *Loong64Parser) parseAssembly(path string) (map[string][]*loong64Line, map[string]int, map[string]*loong64ConstPool, error) {
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
		functions    = make(map[string][]*loong64Line)
		constPools   = make(map[string]*loong64ConstPool)
		functionName string
		labelName    string
		// Constant pool parsing state
		currentConstPool  *loong64ConstPool
		currentConstLabel string
	)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()

		// Check for constant pool label (.LCPI0_0:)
		if loong64ConstPoolLabel.MatchString(line) {
			// Save previous constant pool if any
			if currentConstPool != nil && len(currentConstPool.Data) > 0 {
				constPools[currentConstLabel] = currentConstPool
			}
			// Start new constant pool
			labelPart := strings.Split(line, ":")[0]
			// Normalize label: strip leading .L to get CPI0_0
			labelPart = strings.TrimPrefix(labelPart, ".L")
			currentConstLabel = labelPart
			currentConstPool = &loong64ConstPool{
				Label: labelPart,
				Data:  make([]uint32, 0),
			}
			continue
		}

		// Parse .word directives for constant pool data (32-bit)
		if currentConstPool != nil {
			if matches := loong64WordDirective.FindStringSubmatch(line); matches != nil {
				val := loong64ParseIntValue(matches[1])
				currentConstPool.Data = append(currentConstPool.Data, uint32(val))
				currentConstPool.Size += 4
				continue
			}
			// Parse .dword directives for constant pool data (64-bit)
			if matches := loong64DwordDirective.FindStringSubmatch(line); matches != nil {
				val := loong64ParseIntValue(matches[1])
				// Store dword as two 32-bit words (little-endian)
				currentConstPool.Data = append(currentConstPool.Data, uint32(val), uint32(val>>32))
				currentConstPool.Size += 8
				continue
			}
		}

		// Check for section change or function start that ends constant pool parsing
		if loong64NameLine.MatchString(line) || strings.HasPrefix(strings.TrimSpace(line), ".section") {
			// Save current constant pool if any
			if currentConstPool != nil && len(currentConstPool.Data) > 0 {
				constPools[currentConstLabel] = currentConstPool
				currentConstPool = nil
				currentConstLabel = ""
			}
		}

		if loong64AttributeLine.MatchString(line) {
			continue
		} else if loong64NameLine.MatchString(line) {
			functionName = strings.Split(line, ":")[0]
			functions[functionName] = make([]*loong64Line, 0)
		} else if loong64LabelLine.MatchString(line) {
			labelName = strings.Split(line, ":")[0]
			labelName = labelName[1:]
			lines := functions[functionName]
			if len(lines) == 1 || lines[len(lines)-1].Assembly != "" {
				functions[functionName] = append(functions[functionName], &loong64Line{Labels: []string{labelName}})
			} else {
				lines[len(lines)-1].Labels = append(lines[len(lines)-1].Labels, labelName)
			}
		} else if loong64CodeLine.MatchString(line) {
			asm := strings.Split(line, "//")[0]
			asm = strings.TrimSpace(asm)
			if labelName == "" {
				functions[functionName] = append(functions[functionName], &loong64Line{Assembly: asm})
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

// loong64ParseIntValue parses a decimal or hex integer value from a string
func loong64ParseIntValue(s string) uint64 {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "0x") || strings.HasPrefix(s, "0X") {
		val, _ := strconv.ParseUint(s[2:], 16, 64)
		return val
	}
	val, _ := strconv.ParseUint(s, 10, 64)
	return val
}

func (p *Loong64Parser) parseObjectDump(dump string, functions map[string][]*loong64Line) error {
	var (
		functionName string
		lineNumber   int
	)
	for i, line := range strings.Split(dump, "\n") {
		line = strings.TrimSpace(line)
		if loong64SymbolLine.MatchString(line) {
			functionName = strings.Split(line, "<")[1]
			functionName = strings.Split(functionName, ">")[0]
			lineNumber = 0
		} else if loong64DataLine.MatchString(line) {
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
			if assembly == "nop" {
				continue
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

// loong64GoRegisterName converts LoongArch register names to Go assembly register names
// $a0-$a7 -> R4-R11, $t0-$t8 -> R12-R20, etc.
func loong64GoRegisterName(loongReg string) string {
	if goReg, ok := loong64RegistersAlias[loongReg]; ok {
		return goReg
	}
	// Handle $rN format directly
	if strings.HasPrefix(loongReg, "$r") {
		return "R" + loongReg[2:]
	}
	// Return as-is if not recognized
	return strings.ToUpper(loongReg)
}

func (p *Loong64Parser) generateGoAssembly(t *TranslateUnit, functions []Function, constPools map[string]*loong64ConstPool) error {
	var builder strings.Builder
	builder.WriteString(p.BuildTags())
	t.writeHeader(&builder)

	// Emit DATA/GLOBL directives for constant pools
	if len(constPools) > 0 {
		builder.WriteString("\n// Constant pool data\n")
		for label, pool := range constPools {
			// Emit DATA directive with little-endian byte order
			// Format: DATA symbol<>+offset(SB)/size, $value
			for i, val := range pool.Data {
				builder.WriteString(fmt.Sprintf("DATA %s<>+%d(SB)/4, $0x%08x\n", label, i*4, val))
			}
			// Emit GLOBL directive to define the symbol size
			builder.WriteString(fmt.Sprintf("GLOBL %s<>(SB), RODATA|NOPTR, $%d\n", label, pool.Size))
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
				if fpRegisterCount < len(loong64FPRegisters) {
					if param.Type == "double" {
						builder.WriteString(fmt.Sprintf("\tMOVD %s+%d(FP), %s\n", param.Name, offset, loong64FPRegisters[fpRegisterCount]))
					} else {
						builder.WriteString(fmt.Sprintf("\tMOVF %s+%d(FP), %s\n", param.Name, offset, loong64FPRegisters[fpRegisterCount]))
					}
					fpRegisterCount++
				} else {
					stack = append(stack, lo.Tuple2[int, Parameter]{A: offset, B: param})
				}
			} else {
				if registerCount < len(loong64Registers) {
					builder.WriteString(fmt.Sprintf("\tMOVV %s+%d(FP), %s\n", param.Name, offset, loong64Registers[registerCount]))
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
			builder.WriteString(fmt.Sprintf("\tADDV $-%d, R3\n", frameSize))
			stackoffset := 0
			for i := 0; i < len(stack); i++ {
				builder.WriteString(fmt.Sprintf("\tMOVV %s+%d(FP), R12\n", stack[i].B.Name, frameSize+stack[i].A))
				builder.WriteString(fmt.Sprintf("\tMOVV R12, (%d)(R3)\n", stackoffset))
				if stack[i].B.Pointer {
					stackoffset += 8
				} else {
					stackoffset += supportedTypes[stack[i].B.Type]
				}
			}
		}

		// Convert interface{} lines back to loong64Line
		for _, lineIface := range function.Lines {
			line, ok := lineIface.(*loong64Line)
			if !ok {
				continue
			}

			// Skip pcaddu12i instructions that reference constant pools (they're no longer needed)
			if matches := loong64PcadduConstPool.FindStringSubmatch(line.Assembly); matches != nil {
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

			// Replace ld.w/ld.d instructions that load from constant pools
			if matches := loong64LdConstPool.FindStringSubmatch(line.Assembly); matches != nil {
				destReg := matches[1]
				baseReg := matches[2]
				constLabel := "CPI" + matches[3]
				if _, hasPool := constPools[constLabel]; hasPool {
					// Emit any labels
					for _, label := range line.Labels {
						builder.WriteString(label)
						builder.WriteString(":\n")
					}
					// Emit load address of constant pool into the base register
					builder.WriteString(fmt.Sprintf("\tMOVV $%s<>(SB), %s\n",
						constLabel, loong64GoRegisterName(baseReg)))
					// Emit load from the address
					// Determine if it's a 32-bit or 64-bit load based on the original instruction
					if strings.Contains(line.Assembly, "ld.d") {
						builder.WriteString(fmt.Sprintf("\tMOVV (%s), %s\n",
							loong64GoRegisterName(baseReg), loong64GoRegisterName(destReg)))
					} else {
						// ld.w - 32-bit load
						builder.WriteString(fmt.Sprintf("\tMOVW (%s), %s\n",
							loong64GoRegisterName(baseReg), loong64GoRegisterName(destReg)))
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
					builder.WriteString(fmt.Sprintf("\tADDV $%d, R3\n", frameSize))
				}
				if function.Type != "void" {
					switch function.Type {
					case "int64_t", "long", "_Bool":
						builder.WriteString(fmt.Sprintf("\tMOVV R4, result+%d(FP)\n", offset))
					case "double":
						builder.WriteString(fmt.Sprintf("\tMOVD F0, result+%d(FP)\n", offset))
					case "float":
						builder.WriteString(fmt.Sprintf("\tMOVF F0, result+%d(FP)\n", offset))
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

	// Write file
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
	RegisterParser("loong64", &Loong64Parser{})
}
