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

// ARM64Parser implements ArchParser for ARM64/NEON architecture
type ARM64Parser struct{}

// arm64 regex patterns
var (
	arm64AttributeLine = regexp.MustCompile(`^\s+\..+$`)
	arm64NameLine      = regexp.MustCompile(`^\w+:.+$`)
	// Match labels like .LBB0_2: (Linux) or LBB0_2: (macOS)
	arm64LabelLine = regexp.MustCompile(`^\.?\w+_\d+:.*$`)
	arm64CodeLine  = regexp.MustCompile(`^\s+\w+.*$`)
	// Match jumps to labels with or without leading dot
	arm64JmpLine    = regexp.MustCompile(`^(b|b\.\w{2})\t\.?\w+_\d+$`)
	arm64SymbolLine = regexp.MustCompile(`^\w+\s+<\w+>:$`)
	arm64DataLine   = regexp.MustCompile(`^\w+:\s+\w+\s+.+$`)
	// Match stack frame allocation: "sub sp, sp, #N" (with optional hex suffix)
	arm64StackAllocLine = regexp.MustCompile(`^\s*sub\s+sp,\s*sp,\s*#(\d+)`)
	// Match pre-decrement stack allocation: "stp ... [sp, #-N]!"
	arm64StackPreDecLine = regexp.MustCompile(`\[sp,\s*#-(\d+)\]!`)
)

// arm64 register sets
var (
	arm64Registers     = []string{"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"}
	arm64FPRegisters   = []string{"F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"}
	arm64NeonRegisters = []string{"V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"}
)

// arm64Line represents a single assembly instruction for ARM64
// Binary is string because ARM64 has fixed-width 32-bit instructions
type arm64Line struct {
	Labels   []string
	Assembly string
	Binary   string
}

func (line *arm64Line) String() string {
	var builder strings.Builder
	if arm64JmpLine.MatchString(line.Assembly) {
		splits := strings.Split(line.Assembly, "\t")
		instruction := strings.Map(func(r rune) rune {
			if r == '.' {
				return -1
			}
			return unicode.ToUpper(r)
		}, splits[0])
		// Handle both Linux (.LBB0_5) and macOS (LBB0_5) label formats
		label := splits[1]
		label = strings.TrimPrefix(label, ".")
		label = strings.TrimPrefix(label, "L")
		builder.WriteString(fmt.Sprintf("%s %s\n", instruction, label))
	} else {
		builder.WriteString("\t")
		builder.WriteString(fmt.Sprintf("WORD $0x%v", line.Binary))
		builder.WriteString("\t// ")
		builder.WriteString(line.Assembly)
		builder.WriteString("\n")
	}
	return builder.String()
}

// Name returns the architecture name
func (p *ARM64Parser) Name() string {
	return "arm64"
}

// BuildTags returns the Go build constraint
func (p *ARM64Parser) BuildTags() string {
	return "//go:build !noasm && arm64\n"
}

// BuildTarget returns the clang target triple
func (p *ARM64Parser) BuildTarget(goos string) string {
	if goos == "darwin" {
		return "arm64-apple-darwin"
	}
	return "arm64-linux-gnu"
}

// CompilerFlags returns architecture-specific compiler flags
func (p *ARM64Parser) CompilerFlags() []string {
	// ARM64 requires x18 to be reserved (platform register on some OSes)
	return []string{"-ffixed-x18"}
}

// Prologue returns C parser prologue for ARM64 NEON types
func (p *ARM64Parser) Prologue() string {
	var prologue strings.Builder
	// Define GOAT_PARSER to skip includes during parsing
	prologue.WriteString("#define GOAT_PARSER 1\n")

	// Define __bf16 for arm_bf16.h (compiler built-in type)
	prologue.WriteString("typedef short __bf16;\n")
	// Define __fp16 for arm_fp16.h
	prologue.WriteString("typedef short __fp16;\n")

	// Define NEON 64-bit vector types
	prologue.WriteString("typedef struct { char _[8]; } int8x8_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } int16x4_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } int32x2_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } int64x1_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } uint8x8_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } uint16x4_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } uint32x2_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } uint64x1_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } float32x2_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } float64x1_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } float16x4_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } bfloat16x4_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } poly8x8_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } poly16x4_t;\n")
	prologue.WriteString("typedef struct { char _[8]; } poly64x1_t;\n")

	// Define NEON 128-bit vector types
	prologue.WriteString("typedef struct { char _[16]; } int8x16_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } int16x8_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } int32x4_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } int64x2_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } uint8x16_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } uint16x8_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } uint32x4_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } uint64x2_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } float32x4_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } float64x2_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } float16x8_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } bfloat16x8_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } poly8x16_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } poly16x8_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } poly64x2_t;\n")
	prologue.WriteString("typedef struct { char _[16]; } poly128_t;\n")

	// Define NEON 128-bit array types (x2, x3, x4)
	for _, base := range []string{"int8x16", "int16x8", "int32x4", "int64x2",
		"uint8x16", "uint16x8", "uint32x4", "uint64x2",
		"float32x4", "float64x2", "float16x8", "bfloat16x8",
		"poly8x16", "poly16x8", "poly64x2"} {
		prologue.WriteString(fmt.Sprintf("typedef struct { char _[32]; } %sx2_t;\n", base))
		prologue.WriteString(fmt.Sprintf("typedef struct { char _[48]; } %sx3_t;\n", base))
		prologue.WriteString(fmt.Sprintf("typedef struct { char _[64]; } %sx4_t;\n", base))
	}

	// Define NEON 64-bit array types (x2, x3, x4)
	for _, base := range []string{"int8x8", "int16x4", "int32x2", "int64x1",
		"uint8x8", "uint16x4", "uint32x2", "uint64x1",
		"float32x2", "float64x1", "float16x4", "bfloat16x4",
		"poly8x8", "poly16x4", "poly64x1"} {
		prologue.WriteString(fmt.Sprintf("typedef struct { char _[16]; } %sx2_t;\n", base))
		prologue.WriteString(fmt.Sprintf("typedef struct { char _[24]; } %sx3_t;\n", base))
		prologue.WriteString(fmt.Sprintf("typedef struct { char _[32]; } %sx4_t;\n", base))
	}

	// Add SVE/SME types and intrinsic stubs from arm64_sve.go
	prologue.WriteString(SVEPrologue())

	return prologue.String()
}

// TranslateAssembly implements the full translation pipeline for ARM64
func (p *ARM64Parser) TranslateAssembly(t *TranslateUnit, functions []Function) error {
	// Parse assembly
	assembly, stackSizes, err := p.parseAssembly(t.Assembly, t.TargetOS)
	if err != nil {
		return err
	}

	// Run objdump
	dump, err := runCommand("objdump", "-d", t.Object)
	if err != nil {
		return err
	}

	// Parse object dump
	if err := p.parseObjectDump(dump, assembly, t.TargetOS); err != nil {
		return err
	}

	// Auto-transform SVE/SME functions (inject smstart/smstop, fix forbidden instructions)
	for fnName, lines := range assembly {
		assembly[fnName] = TransformSVEFunction(lines)
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

	// Generate Go assembly
	return p.generateGoAssembly(t, functions)
}

func (p *ARM64Parser) parseAssembly(path string, targetOS string) (map[string][]*arm64Line, map[string]int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer func(file *os.File) {
		if err = file.Close(); err != nil {
			_, _ = fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}(file)

	var (
		stackSizes   = make(map[string]int)
		functions    = make(map[string][]*arm64Line)
		functionName string
		labelName    string
	)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if arm64AttributeLine.MatchString(line) {
			continue
		} else if arm64LabelLine.MatchString(line) {
			// Check labels BEFORE function names because labels like "LBB0_2: ; comment"
			// can match the function name pattern due to content after the colon
			labelName = strings.Split(line, ":")[0]
			// Strip leading dot and L prefix (Linux uses .LBB0_2, macOS uses LBB0_2)
			labelName = strings.TrimPrefix(labelName, ".")
			labelName = strings.TrimPrefix(labelName, "L")
			lines := functions[functionName]
			if len(lines) == 0 || lines[len(lines)-1].Assembly != "" {
				functions[functionName] = append(functions[functionName], &arm64Line{Labels: []string{labelName}})
			} else {
				lines[len(lines)-1].Labels = append(lines[len(lines)-1].Labels, labelName)
			}
		} else if arm64NameLine.MatchString(line) {
			functionName = strings.Split(line, ":")[0]
			// On macOS, function names are prefixed with underscore - strip it
			if targetOS == "darwin" && strings.HasPrefix(functionName, "_") {
				functionName = functionName[1:]
			}
			functions[functionName] = make([]*arm64Line, 0)
			labelName = ""
		} else if arm64CodeLine.MatchString(line) {
			asm := strings.Split(line, "//")[0]
			asm = strings.TrimSpace(asm)

			// Detect stack frame allocation patterns:
			// 1. "sub sp, sp, #N" - explicit stack allocation
			// 2. "stp ... [sp, #-N]!" - pre-decrement stack allocation
			// Record the maximum stack size for this function
			if matches := arm64StackAllocLine.FindStringSubmatch(asm); matches != nil {
				if size, err := strconv.Atoi(matches[1]); err == nil {
					if current, ok := stackSizes[functionName]; !ok || size > current {
						stackSizes[functionName] = size
					}
				}
			} else if matches := arm64StackPreDecLine.FindStringSubmatch(asm); matches != nil {
				if size, err := strconv.Atoi(matches[1]); err == nil {
					if current, ok := stackSizes[functionName]; !ok || size > current {
						stackSizes[functionName] = size
					}
				}
			}

			if labelName == "" {
				functions[functionName] = append(functions[functionName], &arm64Line{Assembly: asm})
			} else {
				lines := functions[functionName]
				if len(lines) > 0 {
					lines[len(lines)-1].Assembly = asm
				}
				labelName = ""
			}
		}
	}

	if err = scanner.Err(); err != nil {
		return nil, nil, err
	}
	return functions, stackSizes, nil
}

func (p *ARM64Parser) parseObjectDump(dump string, functions map[string][]*arm64Line, targetOS string) error {
	var (
		functionName string
		lineNumber   int
	)
	for i, line := range strings.Split(dump, "\n") {
		line = strings.TrimSpace(line)
		if arm64SymbolLine.MatchString(line) {
			functionName = strings.Split(line, "<")[1]
			functionName = strings.Split(functionName, ">")[0]
			// On macOS, function names are prefixed with underscore - strip it
			if targetOS == "darwin" && strings.HasPrefix(functionName, "_") {
				functionName = functionName[1:]
			}
			lineNumber = 0
		} else if arm64DataLine.MatchString(line) {
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

func (p *ARM64Parser) generateGoAssembly(t *TranslateUnit, functions []Function) error {
	var builder strings.Builder
	builder.WriteString(p.BuildTags())
	t.writeHeader(&builder)

	for _, function := range functions {
		// Calculate return size based on type
		returnSize := 0
		if function.Type != "void" {
			if sz := NeonTypeSize(function.Type); sz > 0 {
				returnSize = sz // Use actual NEON type size
			} else if sz, ok := supportedTypes[function.Type]; ok {
				returnSize = sz // Use actual scalar type size
			} else {
				returnSize = 8 // Default 8-byte slot for pointers/unknown types
			}
		}

		registerCount, fpRegisterCount, neonRegisterCount, offset := 0, 0, 0, 0
		var stack []lo.Tuple2[int, Parameter]
		var argsBuilder strings.Builder

		for _, param := range function.Parameters {
			// Calculate slot size based on type
			sz := 8 // Default 8-byte slot
			if !param.Pointer {
				if neonSz := NeonTypeSize(param.Type); neonSz > 0 {
					sz = neonSz // Use actual NEON type size
				} else if param.Type == "float" {
					sz = 4 // float32 is 4 bytes
				}
				// double, int64_t, long, pointers use default 8 bytes
			}
			// Go's ABI uses 8-byte alignment for stack parameters, regardless of type.
			// The natural alignment of SIMD types is a hardware concern handled by registers,
			// not by padding the stack frame.
			alignTo := 8
			if sz < 8 {
				alignTo = sz // Smaller types can use their natural alignment
			}
			// Align offset
			if offset%alignTo != 0 {
				offset += alignTo - offset%alignTo
			}

			if !param.Pointer && IsNeonType(param.Type) {
				// NEON vector type - load into V register(s)
				vecCount := NeonVectorCount(param.Type)
				is64bit := IsNeon64Type(param.Type)

				if neonRegisterCount+vecCount <= len(arm64NeonRegisters) {
					for v := 0; v < vecCount; v++ {
						vecOffset := offset + v*16
						if is64bit {
							vecOffset = offset + v*8
						}

						if is64bit {
							// 64-bit vector: single MOVD, load into D[0] only
							argsBuilder.WriteString(fmt.Sprintf("\tMOVD %s+%d(FP), R9\n", param.Name, vecOffset))
							argsBuilder.WriteString(fmt.Sprintf("\tVMOV R9, %s.D[0]\n", arm64NeonRegisters[neonRegisterCount+v]))
						} else {
							// 128-bit vector: two MOVDs, load into D[0] and D[1]
							// Use _N suffixes (N=offset within param) so go vet accepts different offsets
							localOffset := v * 16 // offset within this parameter's storage
							argsBuilder.WriteString(fmt.Sprintf("\tMOVD %s_%d+%d(FP), R9\n", param.Name, localOffset, vecOffset))
							argsBuilder.WriteString(fmt.Sprintf("\tMOVD %s_%d+%d(FP), R10\n", param.Name, localOffset+8, vecOffset+8))
							argsBuilder.WriteString(fmt.Sprintf("\tVMOV R9, %s.D[0]\n", arm64NeonRegisters[neonRegisterCount+v]))
							argsBuilder.WriteString(fmt.Sprintf("\tVMOV R10, %s.D[1]\n", arm64NeonRegisters[neonRegisterCount+v]))
						}
					}
					neonRegisterCount += vecCount
				} else {
					stack = append(stack, lo.Tuple2[int, Parameter]{A: offset, B: param})
				}
			} else if !param.Pointer && (param.Type == "float" || param.Type == "double") {
				if fpRegisterCount < len(arm64FPRegisters) {
					if param.Type == "float" {
						argsBuilder.WriteString(fmt.Sprintf("\tFMOVS %s+%d(FP), %s\n", param.Name, offset, arm64FPRegisters[fpRegisterCount]))
					} else {
						argsBuilder.WriteString(fmt.Sprintf("\tFMOVD %s+%d(FP), %s\n", param.Name, offset, arm64FPRegisters[fpRegisterCount]))
					}
					fpRegisterCount++
				} else {
					stack = append(stack, lo.Tuple2[int, Parameter]{A: offset, B: param})
				}
			} else {
				if registerCount < len(arm64Registers) {
					argsBuilder.WriteString(fmt.Sprintf("\tMOVD %s+%d(FP), %s\n", param.Name, offset, arm64Registers[registerCount]))
					registerCount++
				} else {
					stack = append(stack, lo.Tuple2[int, Parameter]{A: offset, B: param})
				}
			}
			offset += sz
		}

		// Check if 128-bit+ NEON types are used (for stack frame alignment)
		has128BitNeon := false
		for _, param := range function.Parameters {
			if !param.Pointer && IsNeonType(param.Type) && !IsNeon64Type(param.Type) {
				has128BitNeon = true
				break
			}
		}
		if !has128BitNeon && IsNeonType(function.Type) && !IsNeon64Type(function.Type) {
			has128BitNeon = true
		}
		// Note: Don't align offset to 16 bytes here - Go's ABI only requires 8-byte
		// alignment for return values. The 16-byte alignment is only needed for the
		// stack frame (stackOffset), not for the FP-relative parameter offsets.

		stackOffset := 0
		if len(stack) > 0 {
			for i := 0; i < len(stack); i++ {
				if neonSz := NeonTypeSize(stack[i].B.Type); neonSz > 0 {
					// NEON vector: copy all bytes to stack
					is64bit := IsNeon64Type(stack[i].B.Type)
					vecCount := NeonVectorCount(stack[i].B.Type)
					for v := 0; v < vecCount; v++ {
						srcOffset := stack[i].A + v*16
						if is64bit {
							srcOffset = stack[i].A + v*8
						}
						if is64bit {
							// 64-bit vector: single 8-byte copy
							argsBuilder.WriteString(fmt.Sprintf("\tMOVD %s+%d(FP), R8\n", stack[i].B.Name, srcOffset))
							argsBuilder.WriteString(fmt.Sprintf("\tMOVD R8, %d(RSP)\n", stackOffset))
							stackOffset += 8
						} else {
							// 128-bit vector: two 8-byte copies
							// Use _N suffixes (N=offset within param) so go vet accepts different offsets
							localOffset := v * 16 // offset within this parameter's storage
							argsBuilder.WriteString(fmt.Sprintf("\tMOVD %s_%d+%d(FP), R8\n", stack[i].B.Name, localOffset, srcOffset))
							argsBuilder.WriteString(fmt.Sprintf("\tMOVD R8, %d(RSP)\n", stackOffset))
							argsBuilder.WriteString(fmt.Sprintf("\tMOVD %s_%d+%d(FP), R8\n", stack[i].B.Name, localOffset+8, srcOffset+8))
							argsBuilder.WriteString(fmt.Sprintf("\tMOVD R8, %d(RSP)\n", stackOffset+8))
							stackOffset += 16
						}
					}
				} else {
					argsBuilder.WriteString(fmt.Sprintf("\tMOVD %s+%d(FP), R8\n", stack[i].B.Name, stack[i].A))
					argsBuilder.WriteString(fmt.Sprintf("\tMOVD R8, %d(RSP)\n", stackOffset))
					if stack[i].B.Pointer {
						stackOffset += 8
					} else {
						stackOffset += supportedTypes[stack[i].B.Type]
					}
				}
			}
		}

		// Align to 16 bytes only if 128-bit+ NEON types are used
		if has128BitNeon && stackOffset%16 != 0 {
			stackOffset += 16 - stackOffset%16
		}

		// Return value must be 8-byte aligned in Go's ABI
		if offset%8 != 0 {
			offset += 8 - offset%8
		}

		// Use the larger of: calculated stack offset (for parameter spill) or
		// detected stack size (from 'sub sp, sp, #N' in the compiled assembly)
		frameSize := stackOffset
		if function.StackSize > frameSize {
			frameSize = function.StackSize
		}

		builder.WriteString(fmt.Sprintf("\nTEXT ·%v(SB), $%d-%d\n",
			function.Name, frameSize, offset+returnSize))
		builder.WriteString(argsBuilder.String())

		// Convert interface{} lines back to arm64Line
		for _, lineIface := range function.Lines {
			line, ok := lineIface.(*arm64Line)
			if !ok {
				continue
			}
			for _, label := range line.Labels {
				builder.WriteString(label)
				builder.WriteString(":\n")
			}
			if line.Assembly == "ret" {
				if function.Type != "void" {
					switch function.Type {
					case "int64_t", "long", "_Bool":
						builder.WriteString(fmt.Sprintf("\tMOVD R0, result+%d(FP)\n", offset))
					case "double":
						builder.WriteString(fmt.Sprintf("\tFMOVD F0, result+%d(FP)\n", offset))
					case "float":
						builder.WriteString(fmt.Sprintf("\tFMOVS F0, result+%d(FP)\n", offset))
					default:
						// Check for NEON vector return types
						if IsNeonType(function.Type) {
							is64bit := IsNeon64Type(function.Type)
							vecCount := NeonVectorCount(function.Type)
							resultOffset := offset
							for v := 0; v < vecCount; v++ {
								vReg := arm64NeonRegisters[v] // V0, V1, V2, V3...
								if is64bit {
									// 64-bit vector: extract D[0] only
									builder.WriteString(fmt.Sprintf("\tVMOV %s.D[0], R9\n", vReg))
									builder.WriteString(fmt.Sprintf("\tMOVD R9, result+%d(FP)\n", resultOffset))
									resultOffset += 8
								} else {
									// 128-bit vector: extract both D[0] and D[1]
									// Use _N suffixes (N=offset within result) so go vet accepts different offsets
									localOffset := v * 16 // offset within result parameter
									builder.WriteString(fmt.Sprintf("\tVMOV %s.D[0], R9\n", vReg))
									builder.WriteString(fmt.Sprintf("\tVMOV %s.D[1], R10\n", vReg))
									builder.WriteString(fmt.Sprintf("\tMOVD R9, result_%d+%d(FP)\n", localOffset, resultOffset))
									builder.WriteString(fmt.Sprintf("\tMOVD R10, result_%d+%d(FP)\n", localOffset+8, resultOffset+8))
									resultOffset += 16
								}
							}
						} else {
							return fmt.Errorf("unsupported return type: %v", function.Type)
						}
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
	RegisterParser("arm64", &ARM64Parser{})
}
