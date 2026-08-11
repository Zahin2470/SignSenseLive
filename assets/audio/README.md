# Audio assets

Nothing is bundled — the app runs fine (just silently) until you drop
your own files here.

```
assets/audio/
├── music/
│   └── ambient.ogg        looping background bed for any camera app
└── sfx/
    ├── stable.wav          a sign becomes a confident, stable reading (live mode)
    ├── correct.wav         Practice mode: correct match
    ├── wrong.wav           Practice mode: wrong sign
    ├── record_start.wav    motion collect/live: recording started
    ├── record_stop.wav     motion collect/live: recording stopped/saved
    └── capture.wav         collect mode: capture toggled on
```

Any file you don't add is simply skipped — `AudioManager` checks
`Path.is_file()` before touching it. `.ogg` is recommended for the
looping music track, `.wav` for SFX (lowest trigger latency).

Good free sources: freesound.org, opengameart.org, kenney.nl/assets
(check each pack's license — most are CC0 or similarly permissive).
