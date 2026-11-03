const assert = require("node:assert/strict");
const { alertTitle, filterEvents, connectionView, monitoringOrigin, peopleView } = require("./app.js");

const events = [
  { id: "one", track_id: "7", level: "MEDIUM", identity: "unknown", messages: ["Prolonged presence"], factors: [], violence: 0.1, weapon: 0, loitering: 0.8 },
  { id: "two", track_id: "8", level: "HIGH", identity: "unknown", messages: ["Active violence"], factors: ["Sustained aggression"], violence: 0.9, weapon: 0.1, loitering: 0 },
];
const options = { tab: "all", severity: "all", query: "", acknowledged: new Set(["one"]) };
assert.equal(alertTitle(events[0]), "Suspicious loitering detected");
assert.equal(alertTitle({ ...events[1], weapon: 0.8 }), "Armed altercation detected");
assert.deepEqual(filterEvents(events, { ...options, tab: "unread" }).map(e => e.id), ["two"]);
assert.deepEqual(filterEvents(events, { ...options, severity: "MEDIUM" }).map(e => e.id), ["one"]);
assert.deepEqual(filterEvents(events, { ...options, query: "  AGGRESSION " }).map(e => e.id), ["two"]);
assert.deepEqual(filterEvents(events, { ...options, query: "7" }).map(e => e.id), ["one"]);
assert.deepEqual(filterEvents(events, { ...options, severity: "HIGH", query: "presence" }), []);
const live = { updated_at: 995, has_frame: true, preview_mode: "annotated", connection: { status: "running" } };
assert.equal(connectionView(live, false, 1000).inferenceLive, true);
assert.equal(connectionView(live, false, 1010).status, "Inference stale");
assert.equal(connectionView(live, true, 1000).status, "Host disconnected");
const cameraOnly = { updated_at: null, has_frame: true, preview_mode: "camera", connection: { status: "camera_only", camera_updated_at: 999 } };
assert.equal(connectionView(cameraOnly, false, 1000).cameraLive, true);
assert.equal(connectionView(cameraOnly, false, 1000).inferenceLive, false);
assert.equal(connectionView(cameraOnly, false, 1010).cameraLive, false);
assert.equal(connectionView({ connection: { status: "checking" } }, false, 1000).status, "Checking Pi gateway");
const people = [{ track_id: "7", identity: "Alex", level: "NORMAL" }, { track_id: "8", identity: "unknown", level: "LOW" }];
const livePeople = { ...live, persons: people, events: [] };
assert.deepEqual(peopleView(livePeople, false, 1000).persons, people);
assert.equal(peopleView(livePeople, false, 1000).available, true);
assert.deepEqual(peopleView({ ...livePeople, persons: [] }, false, 1000).persons, []);
assert.equal(peopleView({ ...livePeople, persons: [] }, false, 1000).available, true);
for (const [state, failed, now] of [[livePeople, false, 1010], [livePeople, true, 1000], [{ ...cameraOnly, persons: people }, false, 1000], [null, false, 1000]]) {
  assert.deepEqual(peopleView(state, failed, now).persons, []);
  assert.equal(peopleView(state, failed, now).available, false);
}
assert.match(peopleView(cameraOnly, false, 1000).note, /recognition is offline/);
assert.match(peopleView({ ...livePeople, demo: true }, false, 1000).note, /Simulated/);
assert.deepEqual(peopleView({ ...livePeople, persons: [{ ...people[0], identity: "Jordan" }] }, false, 1000).persons.map(person => person.identity), ["Jordan"]);
assert.equal(monitoringOrigin("https://monitor.example/", "https://sentinel.vercel.app", "https:"), "https://monitor.example");
assert.equal(monitoringOrigin("", "http://127.0.0.1:8080", "http:"), "http://127.0.0.1:8080");
for (const address of ["http://192.168.1.50:8080", "https://user:pass@monitor.example", "https://monitor.example/api/state", "https://monitor.example/?token=secret", "ftp://monitor.example", ""]) {
  assert.throws(() => monitoringOrigin(address, "https://sentinel.vercel.app", "https:"));
}
console.log("Frontend checks passed: alerts, filters, live identities, connection state, hosted URLs and mixed-content rejection.");
