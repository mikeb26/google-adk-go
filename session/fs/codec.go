// Copyright 2026 Mike Brown
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package fs

import (
	"encoding/json"

	"gopkg.in/yaml.v3"
)

// Encoding describes how session data is serialized to and from disk.
type Encoding interface {
	// Ext returns the file extension used for persisted files, without a leading dot.
	Ext() string
	Marshal(v any) ([]byte, error)
	Unmarshal(data []byte, v any) error
}

type jsonEncoding struct{}

func (jsonEncoding) Ext() string { return "json" }

func (jsonEncoding) Marshal(v any) ([]byte, error) { return json.Marshal(v) }

func (jsonEncoding) Unmarshal(data []byte, v any) error { return json.Unmarshal(data, v) }

type yamlEncoding struct{}

func (yamlEncoding) Ext() string { return "yaml" }

func (yamlEncoding) Marshal(v any) ([]byte, error) { return yaml.Marshal(v) }

func (yamlEncoding) Unmarshal(data []byte, v any) error { return yaml.Unmarshal(data, v) }

// JSONEncoding returns the built-in JSON encoder/decoder.
func JSONEncoding() Encoding { return jsonEncoding{} }

// YAMLEncoding returns the built-in YAML encoder/decoder.
func YAMLEncoding() Encoding { return yamlEncoding{} }
