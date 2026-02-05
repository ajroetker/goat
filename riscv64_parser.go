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
	var prologue strings.Builder
	// Define include guards so real system intrinsic headers are skipped.
	prologue.WriteString("#define GOAT_PARSER 1\n")
	prologue.WriteString("#define _RISCV_VECTOR_H 1\n")     // riscv_vector.h
	prologue.WriteString("#define __RISCV_VECTOR_H 1\n")     // alternative guard
	return prologue.String()
}

// TranslateAssembly implements the full translation pipeline for RISC-V 64-bit
func (p *RISCV64Parser) TranslateAssembly(t *TranslateUnit, functions []Function) error {
	// Parse assembly
	assembly, stackSizes, err := p.parseAssembly(t.Assembly)
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

	// Generate Go assembly
	return p.generateGoAssembly(t, functions)
}

func (p *RISCV64Parser) parseAssembly(path string) (map[string][]*riscv64Line, map[string]int, error) {
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
		functions    = make(map[string][]*riscv64Line)
		functionName string
		labelName    string
	)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
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

	if err = scanner.Err(); err != nil {
		return nil, nil, err
	}
	return functions, stackSizes, nil
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

func (p *RISCV64Parser) generateGoAssembly(t *TranslateUnit, functions []Function) error {
	// generate code
	var builder strings.Builder
	builder.WriteString(p.BuildTags())
	t.writeHeader(&builder)

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

		// Convert interface{} lines back to riscv64Line
		for _, lineIface := range function.Lines {
			line, ok := lineIface.(*riscv64Line)
			if !ok {
				continue
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
