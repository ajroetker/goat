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

	// Stack management patterns - these need to be removed since Go handles the frame
	// Match "subq $N, %rsp" - stack allocation
	amd64StackAllocLine = regexp.MustCompile(`subq\s+\$(\d+),\s*%rsp`)
	// Match "addq $N, %rsp" - stack deallocation
	amd64StackDeallocLine = regexp.MustCompile(`addq\s+\$(\d+),\s*%rsp`)
	// Match "pushq %rbx" etc - callee-saved register push (rbx, rbp, r12-r15)
	amd64CalleeSavePush = regexp.MustCompile(`pushq\s+%(rbx|rbp|r1[2-5])`)
	// Match "popq %rbx" etc - callee-saved register pop
	amd64CalleeSavePop = regexp.MustCompile(`popq\s+%(rbx|rbp|r1[2-5])`)

	// Constant pool patterns
	// Match constant pool labels: .LCPI0_0: (Linux) or LCPI0_0: (macOS)
	amd64ConstPoolLabel = regexp.MustCompile(`^\.?LCPI\d+_\d+:`)
	// Match .long directive with hex or decimal value
	amd64LongDirective = regexp.MustCompile(`^\s+\.long\s+(0x[0-9a-fA-F]+|\d+)`)
	// Match .quad directive with hex or decimal value
	amd64QuadDirective = regexp.MustCompile(`^\s+\.quad\s+(0x[0-9a-fA-F]+|\d+)`)
	// Match section directive for rodata
	amd64RodataSection = regexp.MustCompile(`^\s+\.section\s+\.rodata|^\s+\.section\s+__TEXT,__const|^\s+\.section\s+__DATA,__const`)
	// Match RIP-relative memory operand referencing constant pool
	// Examples: .LCPI0_0(%rip), LCPI0_0(%rip)
	amd64RIPRelativeConstPool = regexp.MustCompile(`\.?LCPI(\d+_\d+)\(%rip\)`)
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

// amd64ConstPool represents a constant pool entry with its label and data
type amd64ConstPool struct {
	Label string   // e.g., "CPI0_0" (without leading L or .)
	Data  []uint32 // Data as 32-bit words (for .long directives)
	Size  int      // Total size in bytes
}

func (line *amd64Line) String() string {
	var builder strings.Builder
	builder.WriteString("\t")
	if strings.HasPrefix(line.Assembly, "j") {
		// Handle both Linux (jl .LBB0_1) and macOS (jl LBB0_1) jump formats
		fields := strings.Fields(line.Assembly)
		op := fields[0]
		// Strip both . and L prefixes for consistency
		operand := strings.TrimPrefix(fields[1], ".")
		operand = strings.TrimPrefix(operand, "L")
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

// hasConstPoolRef returns true if this instruction references a constant pool via RIP-relative addressing
func (line *amd64Line) hasConstPoolRef() bool {
	return amd64RIPRelativeConstPool.MatchString(line.Assembly)
}

// getConstPoolLabel extracts the constant pool label from an instruction (e.g., "CPI0_0")
func (line *amd64Line) getConstPoolLabel() string {
	matches := amd64RIPRelativeConstPool.FindStringSubmatch(line.Assembly)
	if len(matches) >= 2 {
		return "CPI" + matches[1]
	}
	return ""
}

// shouldSkip returns true if this instruction should be removed from output.
// This handles C-style stack management that conflicts with Go's frame allocation.
// Go allocates the stack frame via the TEXT directive, so we must remove:
// - pushq/popq of callee-saved registers (rbx, rbp, r12-r15)
// - subq $N, %rsp (stack allocation)
// - addq $N, %rsp (stack deallocation)
func (line *amd64Line) shouldSkip() bool {
	asm := line.Assembly

	// Skip callee-saved register push/pop
	if amd64CalleeSavePush.MatchString(asm) || amd64CalleeSavePop.MatchString(asm) {
		return true
	}

	// Skip explicit stack allocation/deallocation
	if amd64StackAllocLine.MatchString(asm) || amd64StackDeallocLine.MatchString(asm) {
		return true
	}

	return false
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
	// Define include guards so real system intrinsic headers are skipped.
	// The modernc.org C parser can't handle GCC/Clang builtins in these headers.
	// All x86 SIMD types are provided as typedefs below instead.
	prologue.WriteString("#define _IMMINTRIN_H_INCLUDED\n")   // GCC immintrin.h
	prologue.WriteString("#define __IMMINTRIN_H 1\n")         // Clang immintrin.h
	prologue.WriteString("#define __AVX512FP16INTRIN_H\n")    // AVX-512 FP16
	prologue.WriteString("#define __AVX512VLFP16INTRIN_H\n")  // AVX-512 VL FP16
	// Define scalar half-precision types for the parser
	prologue.WriteString("typedef unsigned short __bf16;\n")   // BF16 scalar
	prologue.WriteString("typedef unsigned short _Float16;\n") // FP16 scalar
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
	assembly, stackSizes, constPools, err := p.parseAssembly(t.Assembly, t.TargetOS)
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
			functions[i].Lines = make([]any, len(lines))
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

func (p *AMD64Parser) parseAssembly(path string, targetOS string) (map[string][]*amd64Line, map[string]int, map[string]*amd64ConstPool, error) {
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
		functions    = make(map[string][]*amd64Line)
		constPools   = make(map[string]*amd64ConstPool)
		functionName string
		labelName    string
		// Constant pool parsing state
		inRodataSection   bool
		currentConstPool  *amd64ConstPool
		currentConstLabel string
	)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()

		// Check for rodata section
		if amd64RodataSection.MatchString(line) {
			inRodataSection = true
			continue
		}

		// Check for constant pool label (.LCPI0_0: or LCPI0_0:)
		if amd64ConstPoolLabel.MatchString(line) {
			// Save previous constant pool if any
			if currentConstPool != nil && len(currentConstPool.Data) > 0 {
				constPools[currentConstLabel] = currentConstPool
			}
			// Start new constant pool
			labelPart := strings.Split(line, ":")[0]
			// Normalize label: strip leading . and L, keep CPI part
			labelPart = strings.TrimPrefix(labelPart, ".")
			labelPart = strings.TrimPrefix(labelPart, "L")
			currentConstLabel = labelPart
			currentConstPool = &amd64ConstPool{
				Label: labelPart,
				Data:  make([]uint32, 0),
			}
			continue
		}

		// Parse .long directives for constant pool data
		if inRodataSection || currentConstPool != nil {
			if matches := amd64LongDirective.FindStringSubmatch(line); matches != nil {
				if currentConstPool != nil {
					val := amd64ParseIntValue(matches[1])
					currentConstPool.Data = append(currentConstPool.Data, uint32(val))
					currentConstPool.Size += 4
				}
				continue
			}
			if matches := amd64QuadDirective.FindStringSubmatch(line); matches != nil {
				if currentConstPool != nil {
					val := amd64ParseIntValue(matches[1])
					// Store quad as two 32-bit words (little-endian)
					currentConstPool.Data = append(currentConstPool.Data, uint32(val), uint32(val>>32))
					currentConstPool.Size += 8
				}
				continue
			}
		}

		// Check for section change that exits rodata section
		if strings.HasPrefix(strings.TrimSpace(line), ".section") && !amd64RodataSection.MatchString(line) {
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

		if amd64AttributeLine.MatchString(line) {
			continue
		} else if amd64LabelLine.MatchString(line) {
			// Check labels BEFORE function names because labels like "LBB0_2: ; comment"
			// can match the function name pattern due to content after the colon
			labelName = strings.Split(line, ":")[0]
			// Strip leading dot and L prefix (Linux uses .LBB0_2, macOS uses LBB0_2)
			labelName = strings.TrimPrefix(labelName, ".")
			labelName = strings.TrimPrefix(labelName, "L")
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

			// Detect stack allocation: "subq $N, %rsp"
			// Record the maximum stack size for this function so TEXT directive
			// can declare the correct frame size (Go handles frame allocation)
			if matches := amd64StackAllocLine.FindStringSubmatch(asm); matches != nil {
				if size, err := strconv.Atoi(matches[1]); err == nil {
					if current, ok := stackSizes[functionName]; !ok || size > current {
						stackSizes[functionName] = size
					}
				}
			}

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

	// Save any remaining constant pool
	if currentConstPool != nil && len(currentConstPool.Data) > 0 {
		constPools[currentConstLabel] = currentConstPool
	}

	if err = scanner.Err(); err != nil {
		return nil, nil, nil, err
	}
	return functions, stackSizes, constPools, nil
}

// amd64ParseIntValue parses a decimal or hex integer value from a string
func amd64ParseIntValue(s string) uint64 {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "0x") || strings.HasPrefix(s, "0X") {
		val, _ := strconv.ParseUint(s[2:], 16, 64)
		return val
	}
	val, _ := strconv.ParseUint(s, 10, 64)
	return val
}

// amd64RewriteConstPoolRef rewrites an instruction that uses RIP-relative addressing
// to reference a constant pool, replacing it with Go's static-base addressing.
// Returns empty string if the instruction pattern is not recognized.
//
// Common patterns:
//
//	vmovaps .LCPI0_0(%rip), %ymm0  -> VMOVAPS CPI0_0<>(SB), Y0
//	vmovdqa .LCPI0_0(%rip), %xmm1  -> VMOVDQA CPI0_0<>(SB), X1
//	vbroadcastss .LCPI0_0(%rip), %ymm0 -> VBROADCASTSS CPI0_0<>(SB), Y0
//	movq .LCPI0_0(%rip), %xmm0     -> MOVQ CPI0_0<>(SB), X0
//	leaq .LCPI0_0(%rip), %rax      -> LEAQ CPI0_0<>(SB), AX
func amd64RewriteConstPoolRef(asm string, constLabel string) string {
	// Extract instruction mnemonic and operands
	fields := strings.Fields(asm)
	if len(fields) < 2 {
		return ""
	}

	mnemonic := strings.ToUpper(fields[0])
	operands := strings.Join(fields[1:], " ")

	// Split operands by comma
	parts := strings.Split(operands, ",")
	if len(parts) < 2 {
		return ""
	}

	srcOp := strings.TrimSpace(parts[0])
	dstOp := strings.TrimSpace(parts[1])

	// Check if source operand is the RIP-relative constant pool reference
	if !amd64RIPRelativeConstPool.MatchString(srcOp) {
		// Maybe destination is the const pool ref (shouldn't happen for loads, but check)
		return ""
	}

	// Convert destination register to Go assembly format
	goReg := amd64ToGoRegister(dstOp)
	if goReg == "" {
		return ""
	}

	// Build the rewritten instruction
	// Format: MNEMONIC symbol<>(SB), REGISTER
	return fmt.Sprintf("\t%s %s<>(SB), %s\t// %s\n", mnemonic, constLabel, goReg, asm)
}

// amd64ToGoRegister converts an x86-64 AT&T syntax register to Go assembly format
// Examples: %ymm0 -> Y0, %xmm1 -> X1, %zmm2 -> Z2, %rax -> AX, %eax -> AX
func amd64ToGoRegister(reg string) string {
	reg = strings.TrimSpace(reg)
	if !strings.HasPrefix(reg, "%") {
		return ""
	}
	reg = reg[1:] // Remove % prefix

	// ZMM registers (AVX-512)
	if strings.HasPrefix(reg, "zmm") {
		num := reg[3:]
		return "Z" + num
	}
	// YMM registers (AVX)
	if strings.HasPrefix(reg, "ymm") {
		num := reg[3:]
		return "Y" + num
	}
	// XMM registers (SSE)
	if strings.HasPrefix(reg, "xmm") {
		num := reg[3:]
		return "X" + num
	}
	// 64-bit general purpose registers
	switch reg {
	case "rax":
		return "AX"
	case "rbx":
		return "BX"
	case "rcx":
		return "CX"
	case "rdx":
		return "DX"
	case "rsi":
		return "SI"
	case "rdi":
		return "DI"
	case "rbp":
		return "BP"
	case "rsp":
		return "SP"
	case "r8":
		return "R8"
	case "r9":
		return "R9"
	case "r10":
		return "R10"
	case "r11":
		return "R11"
	case "r12":
		return "R12"
	case "r13":
		return "R13"
	case "r14":
		return "R14"
	case "r15":
		return "R15"
	}
	// 32-bit general purpose registers (Go uses same name for 64 and 32 bit access)
	switch reg {
	case "eax":
		return "AX"
	case "ebx":
		return "BX"
	case "ecx":
		return "CX"
	case "edx":
		return "DX"
	case "esi":
		return "SI"
	case "edi":
		return "DI"
	case "ebp":
		return "BP"
	case "esp":
		return "SP"
	case "r8d":
		return "R8"
	case "r9d":
		return "R9"
	case "r10d":
		return "R10"
	case "r11d":
		return "R11"
	case "r12d":
		return "R12"
	case "r13d":
		return "R13"
	case "r14d":
		return "R14"
	case "r15d":
		return "R15"
	}
	return ""
}

func amd64SanitizeAsm(asm string) string {
	asm = strings.TrimSpace(asm)
	asm = strings.Split(asm, "//")[0]
	// Strip ## comments from objdump output (macOS objdump uses ## for inline comments)
	// Go's assembler rejects # unless it's the first item on a line.
	asm = strings.Split(asm, "##")[0]
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

func (p *AMD64Parser) generateGoAssembly(t *TranslateUnit, functions []Function, constPools map[string]*amd64ConstPool) error {
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
			sz := 8 // Default 8-byte slot
			if !param.Pointer {
				if simdSz := X86SIMDTypeSize(param.Type); simdSz > 0 {
					sz = simdSz
				} else if param.Type == "float" {
					sz = 4 // float32 is 4 bytes
				}
			}

			// Go's ABI uses 8-byte alignment for stack parameters on 64-bit systems.
			// Parameters are placed at 8-byte aligned offsets.
			// The natural alignment of SIMD types is a hardware concern handled by registers,
			// not by padding the stack frame.
			alignTo := min(sz,
				// Smaller types can use their natural alignment
				8)
			if offset%alignTo != 0 {
				offset += alignTo - offset%alignTo
			}
			// Frame size uses actual type size (go vet validates this)

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
					// Use appropriate load instruction based on type size
					loadInstr := "MOVQ" // Default 8-byte load
					if !param.Pointer {
						switch sz {
						case 4:
							loadInstr = "MOVL" // 4-byte load (zero-extends to 64-bit)
						case 2:
							loadInstr = "MOVWLZX" // 2-byte load with zero extension
						case 1:
							loadInstr = "MOVBLZX" // 1-byte load with zero extension
						}
					}
					argsBuilder.WriteString(fmt.Sprintf("\t%s %s+%d(FP), %s\n", loadInstr, param.Name, offset, amd64Registers[registerIndex]))
					registerIndex++
				} else {
					stack = append(stack, lo.Tuple2[int, Parameter]{A: offset, B: param})
				}
			}
			offset += sz
		}

		// Align result offset to pointer size (8 bytes) for Go ABI0.
		// Go packs arguments with natural alignment, but the result
		// section starts at the next pointer-aligned boundary.
		if offset%8 != 0 {
			offset += 8 - offset%8
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

		// Overflow args must be placed above the C function's local variables.
		// The C compiler emits code that reads overflow args at [rsp + localFrameSize],
		// so we store them at SP + function.StackSize.
		spillBase := function.StackSize
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

		// The frame must hold both the C function's local variables and the overflow args.
		frameSize := function.StackSize + stackOffset
		// Ensure 16-byte alignment for the frame
		if frameSize > 0 && frameSize%16 != 0 {
			frameSize += 16 - frameSize%16
		}

		builder.WriteString(fmt.Sprintf("\nTEXT ·%v(SB), $%d-%d\n",
			function.Name, frameSize, offset+returnSize))
		builder.WriteString(argsBuilder.String())

		if len(stack) > 0 {
			spillOffset := 0
			for i := 0; i < len(stack); i++ {
				builder.WriteString(fmt.Sprintf("\tMOVQ %s+%d(FP), R11\n", stack[i].B.Name, stack[i].A))
				builder.WriteString(fmt.Sprintf("\tMOVQ R11, %d(SP)\n", spillBase+spillOffset))
				if simdSz := X86SIMDTypeSize(stack[i].B.Type); simdSz > 0 {
					spillOffset += simdSz
				} else if stack[i].B.Pointer {
					spillOffset += 8
				} else {
					spillOffset += supportedTypes[stack[i].B.Type]
				}
			}
		}

		// Convert interface{} lines back to amd64Line
		for _, lineIface := range function.Lines {
			line, ok := lineIface.(*amd64Line)
			if !ok {
				continue
			}

			// Emit labels even for skipped instructions, so jump targets remain valid
			for _, label := range line.Labels {
				builder.WriteString(label)
				builder.WriteString(":\n")
			}

			// Skip C-style stack management instructions (Go handles the frame)
			if line.shouldSkip() {
				continue
			}

			// Handle instructions that reference constant pools
			// RIP-relative addressing won't work, so we replace with static-base loads
			if line.hasConstPoolRef() {
				constLabel := line.getConstPoolLabel()
				if _, hasPool := constPools[constLabel]; hasPool {
					goAsm := amd64RewriteConstPoolRef(line.Assembly, constLabel)
					if goAsm != "" {
						builder.WriteString(goAsm)
						continue
					}
					// If we couldn't rewrite it, emit as comment and continue
					builder.WriteString(fmt.Sprintf("\t// TODO: rewrite const pool ref: %s\n", line.Assembly))
					continue
				}
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
