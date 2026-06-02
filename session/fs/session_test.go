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
	"sync"
	"testing"
	"time"

	"google.golang.org/adk/session"
)

func TestCloneHelpersAndTempDeltaTrim(t *testing.T) {
	t.Helper()

	srcMap := map[string]any{"k": "v"}
	clonedMap := cloneMap(srcMap)
	clonedMap["k"] = "changed"
	if srcMap["k"] != "v" {
		t.Fatalf("cloneMap mutated input map: got %v", srcMap["k"])
	}

	if got := cloneMap(nil); got == nil {
		t.Fatal("cloneMap(nil) returned nil, want empty map")
	}

	events := []*session.Event{{ID: "e1"}}
	clonedEvents := cloneEvents(events)
	if len(clonedEvents) != len(events) || clonedEvents[0] != events[0] {
		t.Fatalf("cloneEvents() mismatch: got %+v, want %+v", clonedEvents, events)
	}
	clonedEvents = append(clonedEvents, &session.Event{ID: "e2"})
	if len(events) != 1 {
		t.Fatalf("cloneEvents() mutated input slice length: got %d, want 1", len(events))
	}

	event := &session.Event{Actions: session.EventActions{StateDelta: map[string]any{
		"temp:discard": 1,
		"user:keep":    2,
	}}}
	trimmed := trimTempDeltaState(event)
	if _, ok := trimmed.Actions.StateDelta["temp:discard"]; ok {
		t.Fatal("trimTempDeltaState kept temp key")
	}
	if got := trimmed.Actions.StateDelta["user:keep"]; got != 2 {
		t.Fatalf("trimTempDeltaState lost non-temp key, got %v", got)
	}

	original := &session.Event{
		ID:                 "e2",
		Timestamp:          time.Unix(1, 2),
		Actions:            session.EventActions{StateDelta: map[string]any{"x": 1}},
		LongRunningToolIDs: []string{"tool1"},
	}
	clonedEvent := cloneEvent(original)
	clonedEvent.Actions.StateDelta["x"] = 99
	clonedEvent.LongRunningToolIDs[0] = "tool2"
	if original.Actions.StateDelta["x"] != 1 {
		t.Fatalf("cloneEvent mutated original state delta: got %v", original.Actions.StateDelta["x"])
	}
	if original.LongRunningToolIDs[0] != "tool1" {
		t.Fatalf("cloneEvent mutated original LongRunningToolIDs: got %q", original.LongRunningToolIDs[0])
	}
}

func TestNewLocalSessionCopiesInputsAndAppendEvent(t *testing.T) {
	t.Helper()

	state := map[string]any{"k": "v"}
	events := []*session.Event{{ID: "e1", Actions: session.EventActions{StateDelta: map[string]any{"x": 1}}}}
	now := time.Date(2026, 1, 2, 3, 4, 5, 6000, time.UTC)
	sess := newLocalSession("app", "user", "session", state, events, now)

	state["k"] = "mutated"
	events[0].ID = "mutated"

	if got := sess.state["k"]; got != "v" {
		t.Fatalf("newLocalSession did not copy state, got %v", got)
	}
	if len(sess.events) != 1 {
		t.Fatalf("newLocalSession copied wrong number of events: got %d, want 1", len(sess.events))
	}
	if sess.events[0] != events[0] {
		t.Fatal("newLocalSession should preserve event pointers")
	}

	if err := sess.appendEvent(&session.Event{
		ID:        "e2",
		Timestamp: now.Add(time.Second),
		Actions: session.EventActions{StateDelta: map[string]any{
			"temp:drop": true,
			"user:keep": "yes",
		}},
	}); err != nil {
		t.Fatalf("appendEvent() error = %v", err)
	}

	if got := sess.state["user:keep"]; got != "yes" {
		t.Fatalf("appendEvent() did not merge state, got %v", got)
	}
	if _, ok := sess.state["temp:drop"]; !ok {
		t.Fatal("appendEvent() removed state from internal session state")
	}
	if got := sess.events[len(sess.events)-1].Actions.StateDelta; len(got) != 1 || got["user:keep"] != "yes" {
		t.Fatalf("appendEvent() did not trim temp keys from persisted event: %+v", got)
	}
	if !sess.updatedAt.Equal(now.Add(time.Second)) {
		t.Fatalf("appendEvent() updatedAt = %v, want %v", sess.updatedAt, now.Add(time.Second))
	}
}

func TestStateGetSet(t *testing.T) {
	t.Helper()

	s := &state{mu: &sync.RWMutex{}, state: map[string]any{"k": "v"}}
	if got, err := s.Get("k"); err != nil || got != "v" {
		t.Fatalf("Get() = (%v, %v), want (v, nil)", got, err)
	}
	if err := s.Set("k2", 42); err != nil {
		t.Fatalf("Set() error = %v", err)
	}
	if got, err := s.Get("k2"); err != nil || got != 42 {
		t.Fatalf("Get() = (%v, %v), want (42, nil)", got, err)
	}
	if _, err := s.Get("missing"); err != session.ErrStateKeyNotExist {
		t.Fatalf("Get() missing error = %v, want %v", err, session.ErrStateKeyNotExist)
	}
}
