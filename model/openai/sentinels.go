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

package openai

import (
	"fmt"

	"github.com/openai/openai-go/v3/shared/constant"
)

func derived[T constant.Constant[T]]() string {
	return fmt.Sprint(constant.ValueOf[T]())
}

var (
	Assistant                          = derived[constant.Assistant]()
	Developer                          = derived[constant.Developer]()
	Model                              = derived[constant.Model]()
	System                             = derived[constant.System]()
	User                               = derived[constant.User]()
	Function                           = derived[constant.Function]()
	FunctionCall                       = derived[constant.FunctionCall]()
	FunctionCallOutput                 = derived[constant.FunctionCallOutput]()
	Error                              = derived[constant.Error]()
	Other                              = derived[constant.Other]()
	InputFile                          = derived[constant.InputFile]()
	InputImage                         = derived[constant.InputImage]()
	InputText                          = derived[constant.InputText]()
	JSONObject                         = derived[constant.JSONObject]()
	JSONSchema                         = derived[constant.JSONSchema]()
	OutputText                         = derived[constant.OutputText]()
	Reasoning                          = derived[constant.Reasoning]()
	ReasoningText                      = derived[constant.ReasoningText]()
	Refusal                            = derived[constant.Refusal]()
	Response                           = derived[constant.Response]()
	ResponseCompleted                  = derived[constant.ResponseCompleted]()
	ResponseFailed                     = derived[constant.ResponseFailed]()
	ResponseIncomplete                 = derived[constant.ResponseIncomplete]()
	ResponseContentPartAdded           = derived[constant.ResponseContentPartAdded]()
	ResponseContentPartDone            = derived[constant.ResponseContentPartDone]()
	ResponseFunctionCallArgumentsDelta = derived[constant.ResponseFunctionCallArgumentsDelta]()
	ResponseFunctionCallArgumentsDone  = derived[constant.ResponseFunctionCallArgumentsDone]()
	ResponseOutputItemAdded            = derived[constant.ResponseOutputItemAdded]()
	ResponseOutputItemDone             = derived[constant.ResponseOutputItemDone]()
	ResponseOutputTextAnnotationAdded  = derived[constant.ResponseOutputTextAnnotationAdded]()
	ResponseOutputTextDelta            = derived[constant.ResponseOutputTextDelta]()
	ResponseOutputTextDone             = derived[constant.ResponseOutputTextDone]()
	ResponseReasoningSummaryTextDelta  = derived[constant.ResponseReasoningSummaryTextDelta]()
	ResponseReasoningSummaryTextDone   = derived[constant.ResponseReasoningSummaryTextDone]()
	ResponseReasoningTextDelta         = derived[constant.ResponseReasoningTextDelta]()
	ResponseReasoningTextDone          = derived[constant.ResponseReasoningTextDone]()
	ResponseRefusalDelta               = derived[constant.ResponseRefusalDelta]()
	ResponseRefusalDone                = derived[constant.ResponseRefusalDone]()
)
