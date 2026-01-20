// Copyright 2025 goat Authors
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
	"testing"
)

func TestShouldSkip_StackAllocation(t *testing.T) {
	// Test stack allocation: subq $496, %rsp
	// Should be skipped (Go handles frame allocation via TEXT directive)
	line := &amd64Line{
		Assembly: "subq\t$496, %rsp",
		Binary:   []string{"48", "81", "ec", "f0", "01", "00", "00"},
	}

	if !line.shouldSkip() {
		t.Error("expected subq $N, %rsp to be skipped")
	}
}

func TestShouldSkip_StackDeallocation(t *testing.T) {
	// Test stack deallocation: addq $496, %rsp
	// Should be skipped
	line := &amd64Line{
		Assembly: "addq\t$496, %rsp",
		Binary:   []string{"48", "81", "c4", "f0", "01", "00", "00"},
	}

	if !line.shouldSkip() {
		t.Error("expected addq $N, %rsp to be skipped")
	}
}

func TestShouldSkip_CalleeSavedPush(t *testing.T) {
	// Test callee-saved register push: pushq %rbx
	// Should be skipped (Go doesn't use C calling convention)
	tests := []struct {
		name     string
		assembly string
	}{
		{"rbx", "pushq\t%rbx"},
		{"rbp", "pushq\t%rbp"},
		{"r12", "pushq\t%r12"},
		{"r13", "pushq\t%r13"},
		{"r14", "pushq\t%r14"},
		{"r15", "pushq\t%r15"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			line := &amd64Line{
				Assembly: tt.assembly,
				Binary:   []string{"53"}, // placeholder
			}
			if !line.shouldSkip() {
				t.Errorf("expected %s to be skipped", tt.assembly)
			}
		})
	}
}

func TestShouldSkip_CalleeSavedPop(t *testing.T) {
	// Test callee-saved register pop: popq %rbx
	// Should be skipped
	tests := []struct {
		name     string
		assembly string
	}{
		{"rbx", "popq\t%rbx"},
		{"rbp", "popq\t%rbp"},
		{"r12", "popq\t%r12"},
		{"r13", "popq\t%r13"},
		{"r14", "popq\t%r14"},
		{"r15", "popq\t%r15"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			line := &amd64Line{
				Assembly: tt.assembly,
				Binary:   []string{"5b"}, // placeholder
			}
			if !line.shouldSkip() {
				t.Errorf("expected %s to be skipped", tt.assembly)
			}
		})
	}
}

func TestShouldSkip_RegularInstruction(t *testing.T) {
	// Test regular instruction: movq %rax, %rbx
	// Should NOT be skipped
	line := &amd64Line{
		Assembly: "movq\t%rax, %rbx",
		Binary:   []string{"48", "89", "c3"},
	}

	if line.shouldSkip() {
		t.Error("regular movq instruction should not be skipped")
	}
}

func TestShouldSkip_VectorInstruction(t *testing.T) {
	// Test vector instruction: vmovups %ymm0, 464(%rsp)
	// Should NOT be skipped (it uses the stack frame for spilling)
	line := &amd64Line{
		Assembly: "vmovups\t%ymm0, 464(%rsp)",
		Binary:   []string{"c5", "fc", "11", "84", "24", "d0", "01", "00", "00"},
	}

	if line.shouldSkip() {
		t.Error("vmovups spill instruction should not be skipped")
	}
}

func TestShouldSkip_NonCalleeSavedPush(t *testing.T) {
	// Test push of non-callee-saved register: pushq %rax
	// Should NOT be skipped (not callee-saved, might be intentional)
	line := &amd64Line{
		Assembly: "pushq\t%rax",
		Binary:   []string{"50"},
	}

	if line.shouldSkip() {
		t.Error("pushq %rax should not be skipped (not callee-saved)")
	}
}
