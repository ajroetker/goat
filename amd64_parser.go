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
	"strings"
	"unicode"

	"github.com/klauspost/asmfmt"
	"github.com/samber/lo"
)

// AMD64Parser implements ArchParser for x86-64 architecture
type AMD64Parser struct{}

// amd64 regex patterns
var (
	amd64AttributeLine = regexp.MustCompile(`^\s+\..+$`)
	amd64NameLine      = regexp.MustCompile(`^\w+:.+$`)
	// Match labels like .LBB0_2: (Linux) or LBB0_2: (macOS)
	amd64LabelLine  = regexp.MustCompile(`^\.?\w+_\d+:.*$`)
	amd64CodeLine   = regexp.MustCompile(`^\s+\w+.+$`)
	amd64SymbolLine = regexp.MustCompile(`^\w+\s+<\w+>:$`)
	amd64DataLine   = regexp.MustCompile(`^\w+:\s+\w+\s+.+$`)
)

// amd64 register sets
var (
	amd64Registers    = []string{"DI", "SI", "DX", "CX", "R8", "R9"}
	amd64XMMRegisters = []string{"X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7"} // 128-bit SSE
	amd64YMMRegisters = []string{"Y0", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7"} // 256-bit AVX
	amd64ZMMRegisters = []string{"Z0", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7"} // 512-bit AVX-512
)

// amd64Line represents a single assembly instruction for AMD64
// Binary is []string because x86 has variable-length instructions
type amd64Line struct {
	Labels   []string
	Assembly string
	Binary   []string
}

func (line *amd64Line) String() string {
	var builder strings.Builder
	builder.WriteString("\t")
	if strings.HasPrefix(line.Assembly, "j") {
		// Handle both Linux (jl .LBB0_1) and macOS (jl LBB0_1) jump formats
		var op, operand string
		if strings.Contains(line.Assembly, ".") {
			// Linux format: jl .LBB0_1 - split by dot
			splits := strings.Split(line.Assembly, ".")
			op = strings.TrimSpace(splits[0])
			operand = splits[1]
		} else {
			// macOS format: jl LBB0_1 - split by whitespace
			fields := strings.Fields(line.Assembly)
			op = fields[0]
			// Label is the second field, strip L prefix to get BB0_1
			operand = strings.TrimPrefix(fields[1], "L")
		}
		builder.WriteString(fmt.Sprintf("%s %s", strings.ToUpper(op), operand))
	} else {
		pos := 0
		for pos < len(line.Binary) {
			if pos > 0 {
				builder.WriteString("; ")
			}
			if len(line.Binary)-pos >= 8 {
				builder.WriteString(fmt.Sprintf("QUAD $0x%v%v%v%v%v%v%v%v",
					line.Binary[pos+7], line.Binary[pos+6], line.Binary[pos+5], line.Binary[pos+4],
					line.Binary[pos+3], line.Binary[pos+2], line.Binary[pos+1], line.Binary[pos]))
				pos += 8
			} else if len(line.Binary)-pos >= 4 {
				builder.WriteString(fmt.Sprintf("LONG $0x%v%v%v%v",
					line.Binary[pos+3], line.Binary[pos+2], line.Binary[pos+1], line.Binary[pos]))
				pos += 4
			} else if len(line.Binary)-pos >= 2 {
				builder.WriteString(fmt.Sprintf("WORD $0x%v%v", line.Binary[pos+1], line.Binary[pos]))
				pos += 2
			} else {
				builder.WriteString(fmt.Sprintf("BYTE $0x%v", line.Binary[pos]))
				pos += 1
			}
		}
		builder.WriteString("\t// ")
		builder.WriteString(line.Assembly)
	}
	builder.WriteString("\n")
	return builder.String()
}

// Name returns the architecture name
func (p *AMD64Parser) Name() string {
	return "amd64"
}

// BuildTags returns the Go build constraint
func (p *AMD64Parser) BuildTags() string {
	return "//go:build !noasm && amd64\n"
}

// BuildTarget returns the clang target triple
func (p *AMD64Parser) BuildTarget(goos string) string {
	if goos == "darwin" {
		return "x86_64-apple-darwin"
	}
	return "x86_64-linux-gnu"
}

// CompilerFlags returns architecture-specific compiler flags
func (p *AMD64Parser) CompilerFlags() []string {
	return nil // AMD64 doesn't need special fixed-register flags
}

// Prologue returns C parser prologue for x86 SIMD types
func (p *AMD64Parser) Prologue() string {
	var prologue strings.Builder
	// Define x86-64 architecture macros for cross-compilation
	// These are needed so x86 intrinsics headers don't error out
	prologue.WriteString("#ifndef __x86_64__\n")
	prologue.WriteString("#define __x86_64__ 1\n")
	prologue.WriteString("#endif\n")
	prologue.WriteString("#ifndef __amd64__\n")
	prologue.WriteString("#define __amd64__ 1\n")
	prologue.WriteString("#endif\n")
	// Define GOAT_PARSER to skip includes during parsing
	prologue.WriteString("#define GOAT_PARSER 1\n")
	// Define x86 SIMD types as opaque structs for the parser
	// SSE (128-bit)
	prologue.WriteString("typedef struct { char _[16]; } __m128;\n")
	prologue.WriteString("typedef struct { char _[16]; } __m128d;\n")
	prologue.WriteString("typedef struct { char _[16]; } __m128i;\n")
	prologue.WriteString("typedef struct { char _[16]; } __m128h;\n")  // FP16
	prologue.WriteString("typedef struct { char _[16]; } __m128bh;\n") // BF16
	// AVX (256-bit)
	prologue.WriteString("typedef struct { char _[32]; } __m256;\n")
	prologue.WriteString("typedef struct { char _[32]; } __m256d;\n")
	prologue.WriteString("typedef struct { char _[32]; } __m256i;\n")
	prologue.WriteString("typedef struct { char _[32]; } __m256h;\n")  // FP16
	prologue.WriteString("typedef struct { char _[32]; } __m256bh;\n") // BF16
	// AVX-512 (512-bit)
	prologue.WriteString("typedef struct { char _[64]; } __m512;\n")
	prologue.WriteString("typedef struct { char _[64]; } __m512d;\n")
	prologue.WriteString("typedef struct { char _[64]; } __m512i;\n")
	prologue.WriteString("typedef struct { char _[64]; } __m512h;\n")  // FP16
	prologue.WriteString("typedef struct { char _[64]; } __m512bh;\n") // BF16
	return prologue.String()
}

// TranslateAssembly implements the full translation pipeline for AMD64
func (p *AMD64Parser) TranslateAssembly(t *TranslateUnit, functions []Function) error {
	// Parse assembly
	assembly, stackSizes, err := p.parseAssembly(t.Assembly, t.TargetOS)
	if err != nil {
		return err
	}

	// Run objdump
	dump, err := runCommand("objdump", "-d", t.Object, "--insn-width", "16")
	if err != nil {
		return err
	}

	// Parse object dump
	if err := p.parseObjectDump(dump, assembly, t.TargetOS); err != nil {
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

	// Generate Go assembly
	return p.generateGoAssembly(t, functions)
}

func (p *AMD64Parser) parseAssembly(path string, targetOS string) (map[string][]*amd64Line, map[string]int, error) {
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
		functions    = make(map[string][]*amd64Line)
		functionName string
		labelName    string
	)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if amd64AttributeLine.MatchString(line) {
			continue
		} else if amd64LabelLine.MatchString(line) {
			// Check labels BEFORE function names because labels like "LBB0_2: ; comment"
			// can match the function name pattern due to content after the colon
			labelName = strings.Split(line, ":")[0]
			// Strip leading dot if present (Linux uses .LBB0_2, macOS uses LBB0_2)
			if strings.HasPrefix(labelName, ".") {
				labelName = labelName[1:]
			}
			lines := functions[functionName]
			if len(lines) > 0 && lines[len(lines)-1].Assembly == "" {
				lines[len(lines)-1].Labels = append(lines[len(lines)-1].Labels, labelName)
			} else {
				functions[functionName] = append(functions[functionName], &amd64Line{Labels: []string{labelName}})
			}
		} else if amd64NameLine.MatchString(line) {
			functionName = strings.Split(line, ":")[0]
			// On macOS, function names are prefixed with underscore - strip it
			if targetOS == "darwin" && strings.HasPrefix(functionName, "_") {
				functionName = functionName[1:]
			}
			functions[functionName] = make([]*amd64Line, 0)
			labelName = ""
		} else if amd64CodeLine.MatchString(line) {
			asm := amd64SanitizeAsm(line)
			if labelName == "" {
				functions[functionName] = append(functions[functionName], &amd64Line{Assembly: asm})
			} else {
				lines := functions[functionName]
				if len(lines) == 0 {
					functions[functionName] = append(functions[functionName], &amd64Line{Labels: []string{labelName}})
					lines = functions[functionName]
				}
				lines[len(lines)-1].Assembly = asm
				labelName = ""
			}
		}
	}

	if err = scanner.Err(); err != nil {
		return nil, nil, err
	}
	return functions, stackSizes, nil
}

func amd64SanitizeAsm(asm string) string {
	asm = strings.TrimSpace(asm)
	asm = strings.Split(asm, "//")[0]
	asm = strings.TrimSpace(asm)
	return asm
}

func (p *AMD64Parser) parseObjectDump(dump string, functions map[string][]*amd64Line, targetOS string) error {
	var (
		functionName string
		lineNumber   int
	)
	for i, line := range strings.Split(dump, "\n") {
		line = strings.TrimSpace(line)
		if amd64SymbolLine.MatchString(line) {
			functionName = strings.Split(line, "<")[1]
			functionName = strings.Split(functionName, ">")[0]
			// On macOS, function names are prefixed with underscore - strip it
			if targetOS == "darwin" && strings.HasPrefix(functionName, "_") {
				functionName = functionName[1:]
			}
			lineNumber = 0
		} else if amd64DataLine.MatchString(line) {
			data := strings.Split(line, ":")[1]
			data = strings.TrimSpace(data)
			splits := strings.Split(data, " ")
			var (
				binary   []string
				assembly string
			)
			for i, s := range splits {
				if s == "" || unicode.IsSpace(rune(s[0])) {
					assembly = strings.Join(splits[i:], " ")
					assembly = strings.TrimSpace(assembly)
					break
				}
				binary = append(binary, s)
			}

			assembly = amd64SanitizeAsm(assembly)
			if strings.Contains(assembly, "nop") {
				continue
			}

			if assembly == "" {
				return fmt.Errorf("try to increase --insn-width of objdump")
			} else if strings.HasPrefix(assembly, "nop") ||
				assembly == "xchg   %ax,%ax" ||
				assembly == "cs nopw 0x0(%rax,%rax,1)" {
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

func (p *AMD64Parser) generateGoAssembly(t *TranslateUnit, functions []Function) error {
	var builder strings.Builder
	builder.WriteString(p.BuildTags())
	t.writeHeader(&builder)

	for _, function := range functions {
		// Calculate return size based on type
		returnSize := 0
		if function.Type != "void" {
			if sz := X86SIMDTypeSize(function.Type); sz > 0 {
				returnSize = sz
			} else if sz, ok := supportedTypes[function.Type]; ok {
				returnSize = sz
			} else {
				returnSize = 8
			}
		}

		registerIndex, xmmRegisterIndex, offset := 0, 0, 0
		var stack []lo.Tuple2[int, Parameter]
		var argsBuilder strings.Builder

		for _, param := range function.Parameters {
			sz := 8
			if !param.Pointer {
				if simdSz := X86SIMDTypeSize(param.Type); simdSz > 0 {
					sz = simdSz
				}
			}

			alignTo := sz
			if alignTo > 16 {
				alignTo = 16
			}
			if offset%alignTo != 0 {
				offset += alignTo - offset%alignTo
			}

			if !param.Pointer && IsX86SIMDType(param.Type) {
				if xmmRegisterIndex < len(amd64XMMRegisters) {
					switch {
					case X86SIMDTypeSize(param.Type) == 64:
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ %s_0+%d(FP), AX\n", param.Name, offset))
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ %s_8+%d(FP), BX\n", param.Name, offset+8))
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ AX, X14\n"))
						argsBuilder.WriteString(fmt.Sprintf("\tPINSRQ $1, BX, X14\n"))
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ %s_16+%d(FP), AX\n", param.Name, offset+16))
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ %s_24+%d(FP), BX\n", param.Name, offset+24))
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ AX, X15\n"))
						argsBuilder.WriteString(fmt.Sprintf("\tPINSRQ $1, BX, X15\n"))
						argsBuilder.WriteString(fmt.Sprintf("\tVINSERTF128 $1, X15, Y14, Y14\n"))
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ %s_32+%d(FP), AX\n", param.Name, offset+32))
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ %s_40+%d(FP), BX\n", param.Name, offset+40))
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ AX, X15\n"))
						argsBuilder.WriteString(fmt.Sprintf("\tPINSRQ $1, BX, X15\n"))
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ %s_48+%d(FP), AX\n", param.Name, offset+48))
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ %s_56+%d(FP), BX\n", param.Name, offset+56))
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ AX, X13\n"))
						argsBuilder.WriteString(fmt.Sprintf("\tPINSRQ $1, BX, X13\n"))
						argsBuilder.WriteString(fmt.Sprintf("\tVINSERTF128 $1, X13, Y15, Y15\n"))
						argsBuilder.WriteString(fmt.Sprintf("\tVINSERTF64X4 $1, Y15, Z14, %s\n", amd64ZMMRegisters[xmmRegisterIndex]))
					case X86SIMDTypeSize(param.Type) == 32:
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ %s_0+%d(FP), AX\n", param.Name, offset))
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ %s_8+%d(FP), BX\n", param.Name, offset+8))
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ AX, X14\n"))
						argsBuilder.WriteString(fmt.Sprintf("\tPINSRQ $1, BX, X14\n"))
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ %s_16+%d(FP), AX\n", param.Name, offset+16))
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ %s_24+%d(FP), BX\n", param.Name, offset+24))
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ AX, X15\n"))
						argsBuilder.WriteString(fmt.Sprintf("\tPINSRQ $1, BX, X15\n"))
						argsBuilder.WriteString(fmt.Sprintf("\tVINSERTF128 $1, X15, Y14, %s\n", amd64YMMRegisters[xmmRegisterIndex]))
					default:
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ %s_0+%d(FP), AX\n", param.Name, offset))
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ %s_8+%d(FP), BX\n", param.Name, offset+8))
						argsBuilder.WriteString(fmt.Sprintf("\tMOVQ AX, %s\n", amd64XMMRegisters[xmmRegisterIndex]))
						argsBuilder.WriteString(fmt.Sprintf("\tPINSRQ $1, BX, %s\n", amd64XMMRegisters[xmmRegisterIndex]))
					}
					xmmRegisterIndex++
				} else {
					stack = append(stack, lo.Tuple2[int, Parameter]{A: offset, B: param})
				}
			} else if !param.Pointer && (param.Type == "double" || param.Type == "float") {
				if xmmRegisterIndex < len(amd64XMMRegisters) {
					if param.Type == "double" {
						argsBuilder.WriteString(fmt.Sprintf("\tMOVSD %s+%d(FP), %s\n", param.Name, offset, amd64XMMRegisters[xmmRegisterIndex]))
					} else {
						argsBuilder.WriteString(fmt.Sprintf("\tMOVSS %s+%d(FP), %s\n", param.Name, offset, amd64XMMRegisters[xmmRegisterIndex]))
					}
					xmmRegisterIndex++
				} else {
					stack = append(stack, lo.Tuple2[int, Parameter]{A: offset, B: param})
				}
			} else {
				if registerIndex < len(amd64Registers) {
					argsBuilder.WriteString(fmt.Sprintf("\tMOVQ %s+%d(FP), %s\n", param.Name, offset, amd64Registers[registerIndex]))
					registerIndex++
				} else {
					stack = append(stack, lo.Tuple2[int, Parameter]{A: offset, B: param})
				}
			}
			offset += sz
		}

		hasSIMD := false
		for _, param := range function.Parameters {
			if !param.Pointer && IsX86SIMDType(param.Type) {
				hasSIMD = true
				break
			}
		}
		if !hasSIMD && IsX86SIMDType(function.Type) {
			hasSIMD = true
		}

		stackOffset := 0
		if len(stack) > 0 {
			for i := 0; i < len(stack); i++ {
				if simdSz := X86SIMDTypeSize(stack[i].B.Type); simdSz > 0 {
					stackOffset += simdSz
				} else if stack[i].B.Pointer {
					stackOffset += 8
				} else {
					stackOffset += supportedTypes[stack[i].B.Type]
				}
			}
		}
		if hasSIMD && stackOffset%16 != 0 {
			stackOffset += 16 - stackOffset%16
		}

		builder.WriteString(fmt.Sprintf("\nTEXT ·%v(SB), $%d-%d\n",
			function.Name, stackOffset, offset+returnSize))
		builder.WriteString(argsBuilder.String())

		if len(stack) > 0 {
			for i := len(stack) - 1; i >= 0; i-- {
				builder.WriteString(fmt.Sprintf("\tPUSHQ %s+%d(FP)\n", stack[i].B.Name, stack[i].A))
			}
			builder.WriteString("\tPUSHQ $0\n")
		}

		// Convert interface{} lines back to amd64Line
		for _, lineIface := range function.Lines {
			line, ok := lineIface.(*amd64Line)
			if !ok {
				continue
			}
			for _, label := range line.Labels {
				builder.WriteString(label)
				builder.WriteString(":\n")
			}
			if line.Assembly == "retq" {
				if len(stack) > 0 {
					for i := 0; i <= len(stack); i++ {
						builder.WriteString("\tPOPQ DI\n")
					}
				}
				if function.Type != "void" {
					switch function.Type {
					case "int64_t", "long", "_Bool":
						builder.WriteString(fmt.Sprintf("\tMOVQ AX, result+%d(FP)\n", offset))
					case "double":
						builder.WriteString(fmt.Sprintf("\tMOVSD X0, result+%d(FP)\n", offset))
					case "float":
						builder.WriteString(fmt.Sprintf("\tMOVSS X0, result+%d(FP)\n", offset))
					default:
						if IsX86SIMDType(function.Type) {
							resultOffset := offset
							switch X86SIMDTypeSize(function.Type) {
							case 64:
								builder.WriteString("\tVEXTRACTF64X4 $0, Z0, Y14\n")
								builder.WriteString("\tVEXTRACTF64X4 $1, Z0, Y15\n")
								builder.WriteString("\tVEXTRACTF128 $0, Y14, X14\n")
								builder.WriteString("\tMOVQ X14, AX\n")
								builder.WriteString("\tPEXTRQ $1, X14, BX\n")
								builder.WriteString(fmt.Sprintf("\tMOVQ AX, result_0+%d(FP)\n", resultOffset))
								builder.WriteString(fmt.Sprintf("\tMOVQ BX, result_8+%d(FP)\n", resultOffset+8))
								builder.WriteString("\tVEXTRACTF128 $1, Y14, X14\n")
								builder.WriteString("\tMOVQ X14, AX\n")
								builder.WriteString("\tPEXTRQ $1, X14, BX\n")
								builder.WriteString(fmt.Sprintf("\tMOVQ AX, result_16+%d(FP)\n", resultOffset+16))
								builder.WriteString(fmt.Sprintf("\tMOVQ BX, result_24+%d(FP)\n", resultOffset+24))
								builder.WriteString("\tVEXTRACTF128 $0, Y15, X15\n")
								builder.WriteString("\tMOVQ X15, AX\n")
								builder.WriteString("\tPEXTRQ $1, X15, BX\n")
								builder.WriteString(fmt.Sprintf("\tMOVQ AX, result_32+%d(FP)\n", resultOffset+32))
								builder.WriteString(fmt.Sprintf("\tMOVQ BX, result_40+%d(FP)\n", resultOffset+40))
								builder.WriteString("\tVEXTRACTF128 $1, Y15, X15\n")
								builder.WriteString("\tMOVQ X15, AX\n")
								builder.WriteString("\tPEXTRQ $1, X15, BX\n")
								builder.WriteString(fmt.Sprintf("\tMOVQ AX, result_48+%d(FP)\n", resultOffset+48))
								builder.WriteString(fmt.Sprintf("\tMOVQ BX, result_56+%d(FP)\n", resultOffset+56))
							case 32:
								builder.WriteString("\tVEXTRACTF128 $0, Y0, X14\n")
								builder.WriteString("\tMOVQ X14, AX\n")
								builder.WriteString("\tPEXTRQ $1, X14, BX\n")
								builder.WriteString(fmt.Sprintf("\tMOVQ AX, result_0+%d(FP)\n", resultOffset))
								builder.WriteString(fmt.Sprintf("\tMOVQ BX, result_8+%d(FP)\n", resultOffset+8))
								builder.WriteString("\tVEXTRACTF128 $1, Y0, X14\n")
								builder.WriteString("\tMOVQ X14, AX\n")
								builder.WriteString("\tPEXTRQ $1, X14, BX\n")
								builder.WriteString(fmt.Sprintf("\tMOVQ AX, result_16+%d(FP)\n", resultOffset+16))
								builder.WriteString(fmt.Sprintf("\tMOVQ BX, result_24+%d(FP)\n", resultOffset+24))
							case 16:
								builder.WriteString("\tMOVQ X0, AX\n")
								builder.WriteString("\tPEXTRQ $1, X0, BX\n")
								builder.WriteString(fmt.Sprintf("\tMOVQ AX, result_0+%d(FP)\n", resultOffset))
								builder.WriteString(fmt.Sprintf("\tMOVQ BX, result_8+%d(FP)\n", resultOffset+8))
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
	RegisterParser("amd64", &AMD64Parser{})
}
