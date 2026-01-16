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

import "fmt"

// Note: Each architecture parser defines its own Line type internally.
// AMD64 uses Binary []string, others use Binary string.
// The parsers handle this internally and don't expose Line in the interface.

// ArchParser defines the interface for architecture-specific parsing and code generation.
type ArchParser interface {
	// Name returns the architecture name (e.g., "amd64", "arm64")
	Name() string

	// BuildTags returns the Go build constraint for generated files
	BuildTags() string

	// BuildTarget returns the clang target triple for the given OS
	BuildTarget(goos string) string

	// CompilerFlags returns architecture-specific compiler flags (e.g., -ffixed-x18 for arm64)
	CompilerFlags() []string

	// Prologue returns C parser prologue with architecture-specific type definitions
	Prologue() string

	// TranslateAssembly parses assembly, objdump, and generates Go assembly.
	// This encapsulates the full translation pipeline so Line types stay internal.
	TranslateAssembly(t *TranslateUnit, functions []Function) error
}

// parsers holds the registered architecture parsers
var parsers = map[string]ArchParser{}

// RegisterParser registers an architecture parser
func RegisterParser(arch string, p ArchParser) {
	parsers[arch] = p
}

// GetParser returns the parser for the given architecture
func GetParser(arch string) (ArchParser, error) {
	if p, ok := parsers[arch]; ok {
		return p, nil
	}
	return nil, fmt.Errorf("unsupported architecture: %s (available: amd64, arm64, loong64, riscv64)", arch)
}

// ListArchitectures returns a list of supported architectures
func ListArchitectures() []string {
	archs := make([]string, 0, len(parsers))
	for arch := range parsers {
		archs = append(archs, arch)
	}
	return archs
}
