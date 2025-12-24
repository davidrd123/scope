# Console Testing Commands

After "Data channel opened" appears in browser console:

```javascript
// Listen for responses (run this first)
window.dataChannel.addEventListener("message", (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type?.includes("response")) console.log("📨", msg);
});

// Snapshot
window.dataChannel.send('{"type": "snapshot_request"}');

// Restore (use snapshot_id from response)
window.dataChannel.send('{"type": "restore_snapshot", "snapshot_id": "YOUR_ID_HERE"}');

// Pause
window.dataChannel.send('{"paused": true}');

// Step (generate one chunk while paused)
window.dataChannel.send('{"type": "step"}');

// Resume
window.dataChannel.send('{"paused": false}');
```
