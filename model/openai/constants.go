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

const (
	defaultBaseURL = "https://api.openai.com/v1"

	streamErrorCode = "STREAM_ERROR"

	customMetadataResponseID = "openai_response_id"
	customMetadataModel      = "openai_model"
	customMetadataStatus     = "openai_status"
	customMetadataPhase      = "openai_phase"

	customMetadataOpenAIReasoningContinuation = "openai_reasoning_continuation"

	defaultResponseSchemaName = "response_schema"

	applicationJSONMIMEType        = "application/json"
	applicationPDFMIMEType         = "application/pdf"
	applicationOctetStreamMIMEType = "application/octet-stream"
	imageMIMETypePrefix            = "image/"
	textCSVMIMEType                = "text/csv"
	textPlainMIMEType              = "text/plain"
	defaultInputFilename           = "input"
	defaultPDFFilename             = "input.pdf"
	defaultCSVFilename             = "input.csv"
	defaultTextFilename            = "input.txt"
	mapKeyContent                  = "content"
	mapKeyName                     = "name"
	mapKeyOutput                   = "output"
	mapKeyProperties               = "properties"
	mapKeyResult                   = "result"
	mapKeyType                     = "type"
	schemaObjectType               = "object"
	openAIOutputMessageIDPrefix    = "msg_adk_model_output_"
	finishReasonMaxOutputTokens    = "max_output_tokens"
	finishReasonMaxTokens          = "max_tokens"
	finishReasonLength             = "length"
	finishReasonContentFilter      = "content_filter"
	finishReasonFailed             = "failed"
	functionResponseToolName       = "function"
	defaultResponseMessage         = "Handle the request as specified in the instructions."
)
