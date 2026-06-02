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
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"google.golang.org/adk/session"
)

type testEncoding struct{ ext string }

func (e testEncoding) Ext() string                   { return e.ext }
func (testEncoding) Marshal(v any) ([]byte, error)   { return json.Marshal(v) }
func (testEncoding) Unmarshal(b []byte, v any) error { return json.Unmarshal(b, v) }

func TestNewSessionServiceValidation(t *testing.T) {
	t.Helper()

	if _, err := NewSessionService("   "); err == nil {
		t.Fatal("NewSessionService() with blank dir returned nil error")
	}

	if _, err := NewSessionService(t.TempDir(), WithEncoding(testEncoding{ext: ""})); err == nil {
		t.Fatal("NewSessionService() with empty encoding extension returned nil error")
	}

	dir := filepath.Join(t.TempDir(), "nested", "sessions")
	svc, err := NewSessionService(dir, WithEncoding(nil))
	if err != nil {
		t.Fatalf("NewSessionService() error = %v", err)
	}
	impl, ok := svc.(*service)
	if !ok {
		t.Fatalf("NewSessionService() returned %T, want *service", svc)
	}
	if impl.encoding.Ext() != "json" {
		t.Fatalf("default encoding ext = %q, want %q", impl.encoding.Ext(), "json")
	}
	if _, err := os.Stat(dir); err != nil {
		t.Fatalf("expected service directory to exist: %v", err)
	}
}

func TestServiceSaveLoadSessionRoundTrip(t *testing.T) {
	t.Helper()

	dir := t.TempDir()
	svc := &service{dir: dir, encoding: JSONEncoding()}
	now := time.Date(2026, 2, 3, 4, 5, 6, 123456000, time.UTC)
	want := &persistedSession{
		AppName:   "app/one",
		UserID:    "user/../two",
		SessionID: "sess.3",
		State: map[string]any{
			"k": "v",
		},
		Events: []*session.Event{{
			ID:        "event-1",
			Timestamp: now,
			Actions: session.EventActions{
				StateDelta: map[string]any{"answer": 42},
			},
		}},
		CreateTime: now,
		UpdateTime: now,
	}

	if err := svc.saveSession(want); err != nil {
		t.Fatalf("saveSession() error = %v", err)
	}

	path := svc.sessionPath(want.AppName, want.UserID, want.SessionID)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected session file at %s: %v", path, err)
	}

	got, err := svc.loadPersistedSession(want.AppName, want.UserID, want.SessionID)
	if err != nil {
		t.Fatalf("loadPersistedSession() error = %v", err)
	}
	if got.AppName != want.AppName || got.UserID != want.UserID || got.SessionID != want.SessionID {
		t.Fatalf("loaded identity mismatch: got %+v, want %+v", got, want)
	}
	if !reflect.DeepEqual(got.State, want.State) {
		t.Fatalf("loaded state mismatch: got %+v, want %+v", got.State, want.State)
	}
	if len(got.Events) != 1 || got.Events[0].ID != want.Events[0].ID || !got.Events[0].Timestamp.Equal(now) {
		t.Fatalf("loaded events mismatch: got %+v, want %+v", got.Events, want.Events)
	}
	if !got.CreateTime.Equal(now) || !got.UpdateTime.Equal(now) {
		t.Fatalf("loaded timestamps mismatch: got create=%v update=%v, want %v", got.CreateTime, got.UpdateTime, now)
	}
}

func TestLoadPersistedSessionMissing(t *testing.T) {
	t.Helper()

	svc := &service{dir: t.TempDir(), encoding: JSONEncoding()}
	_, err := svc.loadPersistedSession("app", "user", "session")
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("loadPersistedSession() error = %v, want fs.ErrNotExist", err)
	}
}

func TestSafeSegmentRoundTrip(t *testing.T) {
	t.Helper()

	cases := []string{"", "plain", "a/b/c", "contains space", "100%", ".", ".."}
	for _, tc := range cases {
		tc := tc
		t.Run(tc, func(t *testing.T) {
			escaped := safeSegment(tc)
			got, err := unsafeSegment(escaped)
			if err != nil {
				t.Fatalf("unsafeSegment(%q) error = %v", escaped, err)
			}
			if got != tc {
				t.Fatalf("round-trip mismatch: got %q, want %q", got, tc)
			}
		})
	}
}

func TestServiceCreateAndDeleteAreCallable(t *testing.T) {
	t.Helper()

	svc := &service{dir: t.TempDir(), encoding: JSONEncoding()}
	_, _ = svc.Create(context.Background(), &session.CreateRequest{AppName: "app", UserID: "user"})
	_ = svc.Delete(context.Background(), &session.DeleteRequest{AppName: "app", UserID: "user", SessionID: "missing"})
}
