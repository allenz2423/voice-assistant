# Live multistep acceptance prompts

These five prompts exercise Adam through its normal voice or text interface. They use only generated fixture data. Run them one at a time in a host where the needed desktop, browser, filesystem, and system-status tools are enabled. Do not use a personal browser profile or personal files.

## Shared setup

Choose a fresh alphanumeric run ID and set `FIXTURE_DIR` to `/tmp/adam-live-acceptance/<RUN_ID>`. Create the fixture; the generator refuses to reuse an existing run ID:

```sh
uv run python tools/create_implementation_fixtures.py \
  --output-root /tmp/adam-live-acceptance \
  --run-id <RUN_ID>
```

For browser scenarios, serve only that fixture directory on loopback and open the report in Adam's isolated browser:

```sh
python3 -m http.server 8766 --bind 127.0.0.1 --directory "$FIXTURE_DIR"
```

Set `<LOCAL_REPORT_URL>` to `http://127.0.0.1:8766/invoice-report.html`. Keep the server bound to loopback. Substitute the actual fixture path for `<FIXTURE_DIR>` and this URL for `<LOCAL_REPORT_URL>` in prompts below. The generator's `expected.json` records independent expected values for the generated run.

## 1. File read plus fresh system status

**Setup:** Create the shared fixture. No browser is needed.

**Exact prompt:**

> Read `<FIXTURE_DIR>/source.txt` and tell me its byte size and exact first line. Then get a fresh system status and report the logical CPU core count and memory-use percentage. Format your final answer as exactly three lines labeled `First line:`, `Bytes:`, and `System:`. Keep the first line's text exact.

**Expected result:** The file's first line is `ITEM: olive`; its byte length is 75. The final answer has three lines and the system line contains current status values.

**Cleanup check:** Confirm the fixture source is unchanged, then remove only this run's fixture directory after all scenarios finish.

## 2. Read the synthetic invoice report

**Setup:** Start the loopback server, then open `<LOCAL_REPORT_URL>` in Adam's isolated browser.

**Exact prompt:**

> In Adam's isolated browser, with the local fixture invoice report at `<LOCAL_REPORT_URL>` already open, find the supplier with the largest total among overdue invoices. Give the supplier, every invoice ID for that supplier's overdue invoices, and the total. Use exactly three labeled lines: `Supplier:`, `Invoice IDs:`, and `Total:`.

**Expected result:** The answer matches `expected.json` → `oracles.invoices` for supplier, invoice IDs, and formatted total. This intentionally requires a multiline final answer.

**Cleanup check:** Close the fixture tab and stop the loopback server with Ctrl-C. Confirm no network-facing server was started.

## 3. Edit and reread a disposable document

**Setup:** Create the shared fixture. The initial document is `<FIXTURE_DIR>/document.txt`.

**Exact prompt:**

> In `<FIXTURE_DIR>/document.txt`, change `DRAFT` to `FINAL` and save it. Leave every other character unchanged, reopen the file, and report its complete saved contents.

**Expected result:** Reopened contents are exactly `STATUS: FINAL\nOWNER: Adam\n`.

**Cleanup check:** Confirm `source.txt` was not changed, then remove this run's fixture directory after the suite.

## 4. Change and verify local fixture settings

**Setup:** Start the loopback server and open `http://127.0.0.1:8766/settings.html` in a disposable browser window. The fixture page resets to Compact off, Sync on, Notifications on when loaded.

**Exact prompt:**

> On the open local fixture settings page, set Compact mode to on, Sync to off, and Notifications to off. Read the page again and tell me the resulting state.

**Expected result:** A fresh read shows `Compact=on; Sync=off; Notifications=off`.

**Cleanup check:** Close the fixture tab and stop the loopback server with Ctrl-C. The settings exist only in the disposable page.

## 5. Stop an active report run

**Setup:** Start the loopback server and open `<LOCAL_REPORT_URL>` in Adam's isolated browser. Ensure `<FIXTURE_DIR>/cancelled-output.txt` does not exist.

**Exact prompt:**

> Inspect all rows in the local invoice report at `<LOCAL_REPORT_URL>` and determine the supplier with the largest overdue total and its invoice IDs. Then tell me the exact line you would write to `<FIXTURE_DIR>/cancelled-output.txt`. Do not create or modify any file; wait for my next instruction.

**Operator action:** As soon as Adam shows that the request is active, use the normal stop/interruption control. If running Adam in the foreground, Ctrl-C may be used to terminate that run. Do not wait for the final answer.

**Expected result:** The run stops promptly and produces no later tool action or response for this request. `cancelled-output.txt` remains absent. Record whether the stop control or process termination was used.

**Cleanup check:** Confirm the output file is absent, close the fixture tab, stop the loopback server with Ctrl-C, and remove only the run's fixture directory.
