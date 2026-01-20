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

func TestTransformStackInstruction_PreIndexedSTP(t *testing.T) {
	// Test pre-indexed STP: stp x28, x27, [sp, #-80]!
	// Original binary: 0xa9bb6ffc
	// Expected: transform to signed-offset stp x28, x27, [sp, #0]
	line := &arm64Line{
		Assembly: "stp\tx28, x27, [sp, #-80]!",
		Binary:   "a9bb6ffc",
	}

	newBinary, transformed := line.transformStackInstruction()
	if !transformed {
		t.Fatal("expected instruction to be transformed")
	}
	if newBinary == "" {
		t.Fatal("expected non-empty binary, got empty (removed)")
	}

	// Verify the transformation:
	// - Bit 23 should be cleared (pre-indexed -> signed offset)
	// - imm7 should be zeroed
	// Original: 1010 1001 1011 1011 0110 1111 1111 1100
	// After:    1010 1001 0000 0000 0110 1111 1111 1100
	expected := "a9006ffc"
	if newBinary != expected {
		t.Errorf("expected binary %s, got %s", expected, newBinary)
	}
}

func TestTransformStackInstruction_PostIndexedLDP(t *testing.T) {
	// Test post-indexed LDP: ldp x28, x27, [sp], #80
	// Original binary: 0xa8c56ffc (post-indexed load)
	// Expected: transform to signed-offset ldp x28, x27, [sp, #0]
	line := &arm64Line{
		Assembly: "ldp\tx28, x27, [sp], #80",
		Binary:   "a8c56ffc",
	}

	newBinary, transformed := line.transformStackInstruction()
	if !transformed {
		t.Fatal("expected instruction to be transformed")
	}
	if newBinary == "" {
		t.Fatal("expected non-empty binary, got empty (removed)")
	}

	// Verify transformation to signed-offset mode
	// Post-indexed has bits 25-23 = 001, signed-offset has 010
	// Set bit 24, clear bit 23, zero imm7
	// L bit (bit 22) = 1 for load, so result is a9406ffc not a9006ffc
	expected := "a9406ffc"
	if newBinary != expected {
		t.Errorf("expected binary %s, got %s", expected, newBinary)
	}
}

func TestTransformStackInstruction_SubSP(t *testing.T) {
	// Test explicit stack allocation: sub sp, sp, #80
	// Should be removed (return empty string)
	line := &arm64Line{
		Assembly: "sub\tsp, sp, #80",
		Binary:   "d10140ff",
	}

	newBinary, transformed := line.transformStackInstruction()
	if !transformed {
		t.Fatal("expected instruction to be transformed")
	}
	if newBinary != "" {
		t.Errorf("expected empty binary (removed), got %s", newBinary)
	}
}

func TestTransformStackInstruction_AddSP(t *testing.T) {
	// Test explicit stack deallocation: add sp, sp, #80
	// Should be removed (return empty string)
	line := &arm64Line{
		Assembly: "add\tsp, sp, #80",
		Binary:   "910140ff",
	}

	newBinary, transformed := line.transformStackInstruction()
	if !transformed {
		t.Fatal("expected instruction to be transformed")
	}
	if newBinary != "" {
		t.Errorf("expected empty binary (removed), got %s", newBinary)
	}
}

func TestTransformStackInstruction_RegularSTP(t *testing.T) {
	// Test regular signed-offset STP: stp x26, x25, [sp, #16]
	// Should NOT be transformed (no pre/post indexing)
	line := &arm64Line{
		Assembly: "stp\tx26, x25, [sp, #16]",
		Binary:   "a90167fa",
	}

	_, transformed := line.transformStackInstruction()
	if transformed {
		t.Error("regular signed-offset STP should not be transformed")
	}
}

func TestTransformStackInstruction_NonStackInstruction(t *testing.T) {
	// Test non-stack instruction: add x0, x1, x2
	// Should NOT be transformed
	line := &arm64Line{
		Assembly: "add\tx0, x1, x2",
		Binary:   "8b020020",
	}

	_, transformed := line.transformStackInstruction()
	if transformed {
		t.Error("non-stack instruction should not be transformed")
	}
}

func TestString_TransformedInstruction(t *testing.T) {
	// Test that String() properly outputs transformed instructions
	line := &arm64Line{
		Assembly: "stp\tx28, x27, [sp, #-80]!",
		Binary:   "a9bb6ffc",
	}

	output := line.String()

	// Should contain the transformed binary
	if output == "" {
		t.Error("expected non-empty output")
	}

	// Should contain [transformed] marker
	if !contains(output, "[transformed]") {
		t.Errorf("expected output to contain [transformed], got: %s", output)
	}

	// Should contain the new binary (a9006ffc)
	if !contains(output, "0xa9006ffc") {
		t.Errorf("expected output to contain transformed binary 0xa9006ffc, got: %s", output)
	}
}

func TestString_RemovedInstruction(t *testing.T) {
	// Test that String() returns empty for removed instructions
	line := &arm64Line{
		Assembly: "sub\tsp, sp, #80",
		Binary:   "d10140ff",
	}

	output := line.String()
	if output != "" {
		t.Errorf("expected empty output for removed instruction, got: %s", output)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
