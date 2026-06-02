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
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestLockFileAndMutexReuse(t *testing.T) {
	t.Helper()

	svc := &service{dir: t.TempDir(), encoding: JSONEncoding()}
	path := filepath.Join(svc.dir, "app", "user", "session.json")

	unlock, err := svc.lockFile(path)
	if err != nil {
		t.Fatalf("lockFile() error = %v", err)
	}
	if _, err := os.Stat(filepath.Join(svc.dir, "app", "user")); err != nil {
		t.Fatalf("expected lock directory to be created: %v", err)
	}
	if _, err := os.Stat(path + lockFileSuffix); err != nil {
		t.Fatalf("expected lock file to exist: %v", err)
	}
	unlock()

	mu1 := svc.getMutex(path + lockFileSuffix)
	mu2 := svc.getMutex(path + lockFileSuffix)
	if mu1 != mu2 {
		t.Fatal("getMutex() returned different mutexes for the same key")
	}

	other := svc.getMutex(filepath.Join(svc.dir, "other") + lockFileSuffix)
	if mu1 == other {
		t.Fatal("getMutex() returned the same mutex for different keys")
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		unlock2, err := svc.lockFile(path)
		if err != nil {
			t.Errorf("second lockFile() error = %v", err)
			return
		}
		unlock2()
	}()
	wg.Wait()
}
