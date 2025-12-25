# Akira Events Guide (from Captions)

This guide summarizes the major narrative beats and visual setpieces described in `akira_video_captions_with_filename.txt`, in rough story order. Use it to jump to sequences and to align captioned clips with key events while testing LoRA checkpoints or building prompt packs.

## Overview
- Opening logos and prologue imagery
- Neo‑Tokyo establishing → Harukiya bar intros
- Capsules vs Clowns biker run (highway chase, Kaneda iconics)
- Police station → school → social texture
- Colonel + Scientists → Hospital containment (Tetsuo)
- Riots and resistance (Kei, Ryu) cross‑cut with Tetsuo’s episodes
- Nursery/Espers visions and psychic breaks (milk, toys, teddy giant)
- Tetsuo escalation → flight from hospital → street confrontations
- SOL orbital laser strike → loss and the metal arm
- Stadium: Akira containers → awakening sequences
- Final confrontation → body‑horror metamorphosis → climax and epilogue

## Timeline (Approximate Order)
- 00:00–00:00:22 — Opening Logos
  - Toho logo/starburst; production cards; fade to black.
  - Example: SDR_proxy_16fps-00.00.00.000–00.00.17.688 (Segments 001–004)

- 00:00:22+ — Neo‑Tokyo Establishing
  - High‑angle push‑downs over layered city; elevated highways; deep urban canyons.
  - Example: SDR_proxy_16fps-00.00.17.688–00.00.27.813 (Segments 005–006)

- Prologue Detonation + Cellular Montage
  - White dome blast overtaking city; cut to microscopic tissue/fluids with web‑like structures.
  - Example: lines around: early “sphere of white light expands” and “microscopic cellular” clips.

- Harukiya Bar Intros (Kaneda, Yamagata, Kai)
  - Neon jukebox; PILL CAPSULE SYMBOL; drink splash; arcade glow.
  - Example: ~00:02:22–00:02:33 (Segments 036–040)

- Capsules vs Clowns (Pre‑Chase Street Confrontations)
  - Alley standoffs; bruised bikers; grimaces; setup before big run.
  - Anchors: “stout biker in pink shirt,” “purple motorcycle,” “graffiti wall” clips.

- Highway Night Run (Bike Iconics)
  - Capsules launch; red/white LIGHT TRAILS; AKIRA SLIDE (later in chase); Clowns pursuit; city glows.
  - Note: file includes numerous bike beats; search for “motorcycle”, “light trails”, “slide”.

- Police Station / School Texture
  - Processing/detention and reprimand vignettes grounding the gang’s world.

- Colonel + Scientists / Hospital Containment (Tetsuo)
  - Sterile labs; pods; medical panels; masked staff; telemetric readouts.

- Riots and Resistance (Kei, Ryu)
  - Tear gas canisters; OPAQUE PAINTED SMOKE PLUMES; handheld‑style crowd tracks; batons; brutality tableaux.
  - Example anchors at lines ~1300+ show “gas canister”, “Ryu tracked through crowd”, “Kei in smoke”.

- Nursery/Espers Sequences
  - Psychic children (Takashi, Kiyoko, Masaru) motifs: toys, milk, flood‑of‑foam; warped perspective illusions.

- Tetsuo Escapes / Street Episodes
  - Hallucinations; rage outbursts; confrontations; Kei/Ryu threads cross back through.

- SOL Orbital Laser
  - Straight‑line beam from sky; surgical strike; massive flare/impact frames; crater aftermath.
  - Followed by: Tetsuo’s improvised metal arm (mechanical grafting setpiece).

- Stadium / Akira Containers
  - Excavation pods; cold storage; labeled canisters; awakening; shockwaves/dust rings.

- Final Confrontation + Metamorphosis
  - Kaneda vs Tetsuo; Kei intervention; psychic energy waves; body horror expansion; Kaori tragedy; engulfment.
  - White genesis field; “I am Tetsuo” cosmic‑creation coda.

## Major Setpieces (Prominent Visual Hooks)
- Biker Run and AKIRA SLIDE
  - Night highway, neon reflections, long light trails, low tracking shots, sparks, smear frames, and the iconic sideways brake.
- Riot Smoke and Street Brutality
  - Tear gas plumes, handheld parallax through crowds, raised fists, batons, and red‑washed ambient haze.
- Nursery Illusions and Toy Monstrosities
  - Milk/medicine floods; stuffed animals growing gigantic; warped, shifting scale within hospital.
- SOL Strike
  - Vertical beam; flash‑white impact frames; crater plume; shock rings; lingering dust in lamplight.
- Stadium Awakening
  - Canister reads; frosted containers; concentric wake patterns when opened; dust‑laden beams.
- Metamorphosis
  - Veined, translucent swell; mechanical + organic fusion; engulfment; layered animated fluids and tissue patterns.

## Character Threads (Quick Reference)
- Kaneda
  - Bar intro; gang lead during runs; visor reflections; direct confrontations at stadium; closing beats.
- Tetsuo
  - Hospital containment → psychic onset → escape → sol strike → metal arm → stadium mutation.
- Kei
  - Resistance contacts; riot sequences; later channel for psychic force; support in the final clash.
- Colonel
  - Military control rooms; helicopter/convoys; SOL authorization; containment orders.
- Espers (Takashi, Kiyoko, Masaru)
  - Nursery; projections; warnings; coordinated interventions in Tetsuo’s path.

## Clip Anchor Examples (Search Aids)
- Harukiya bar: “jukebox”, “PILL CAPSULE SYMBOL”, “drink splash”, “arcade glow”
- Riots: “canister”, “OPAQUE PAINTED SMOKE PLUMES”, “Ryu”, “baton”, “riot gear”
- Bikes: “motorcycle”, “light trails”, “slide”, “pursuit”
- Hospital/Nursery: “milk”, “toy”, “teddy”, “nursery”, “monitor”, “restraints”
- SOL/Stadium: “beam”, “satellite”, “stadium”, “container”, “pods”, “dust ring”
- Metamorphosis: “mutating arm”, “body horror”, “engulf”, “veins”, “translucent”

## Notes for Prompting / LoRA Tests
- Use this sequence as a backbone to select balanced samples across moods: neon/night action, sterile hospital, riot haze, red logic center, stadium dust, white genesis.
- For time anchoring, the filenames encode `HH:MM:SS.mmm` ranges; use those segments to fetch adjacent context.
- When generating, preserve Akira’s optical simulation: rack focus, bokeh, chromatic flares, animated grain, ones for fast strikes and twos for holds.

## Dense Timeline Anchors (Segment Ranges)

- Opening Logos
  - SDR_proxy_16fps-00.00.00.000-00.00.03.438-Segment_001
  - SDR_proxy_16fps-00.00.03.438-00.00.08.500-Segment_002
  - SDR_proxy_16fps-00.00.08.500-00.00.12.625-Segment_003
  - SDR_proxy_16fps-00.00.12.625-00.00.17.688-Segment_004

- Neo‑Tokyo Establishing (city, highways)
  - 00:00:17.688–00:00:37.938 (Segments 005–008)
  - 00:01:03.250–00:01:13.375 (Segments 014–015)

- Prologue Detonation (white dome)
  - 00:00:37.938–00:00:43.000 (Segment 009)

- Cellular Montage (microscopic tissue/fluids)
  - 00:00:48.063–00:01:08.313 (Segments 011–014)

- Harukiya Bar Intros (Kaneda, Yamagata, Kai)
  - 00:02:08.063–00:02:19.750 (Segments 031–034)
  - 00:02:22.750–00:02:33.625 (Segments 036–040)
  - 00:02:39.813–00:02:47.625 (Segments 045–046)
  - 00:02:53.438–00:03:05.000 (Segments 049–052)

- Highway Night Run (Capsules vs Clowns; bike iconics)
  - 00:03:05.000–00:03:56.188 (Segments 053–071)
  - 00:04:15.938–00:04:43.063 (Segments 080–089)
  - 00:04:54.438–00:05:16.063 (Segments 094–102)
  - 00:05:17.688–00:05:41.000 (Segments 103–109)
  - 00:05:44.500–00:06:12.000 (Segments 112–127)
  - 00:06:13.250–00:06:26.375 (Segments 129–134)
  - 00:06:36.688–00:06:56.438 (Segments 140–149)

- Police/School Texture (processing, reprimands)
  - Distributed interludes across ~00:11:30–00:12:12 (e.g., Segments 256–272)

- Colonel + Scientists / Hospital Containment (Tetsuo)
  - Early monitor/lab motifs appear around ~00:10:25–00:11:17 (e.g., Segments 228–250)
  - Additional hospital beats interleave with nursery events (see below)

- Riots and Resistance (Kei, Ryu; smoke, batons)
  - 00:07:03.438–00:07:26.750 (Segments 154–161)
  - 00:07:35.750–00:08:01.875 (Segments 166–176)
  - 00:08:10.375–00:08:27.563 (Segments 180–184)
  - 00:08:37.688–00:08:58.438 (Segments 188–196)
  - 00:09:02.688–00:09:28.250 (Segments 198–206)
  - 00:09:34.938–00:10:06.000 (Segments 209–220)
  - 00:10:10.500–00:10:34.188 (Segments 224–230)
  - 00:10:57.438–00:11:12.313 (Segments 241–247)
  - 00:11:31.688–00:12:01.750 (Segments 257–268)

- Nursery/Espers Sequences (toys, milk, illusions)
  - 00:06:12.000–00:06:13.250 (Segment 128)
  - 00:07:20.500–00:07:26.750 (Segments 159–161)
  - 00:07:43.938–00:07:54.313 (Segments 170–173)
  - 00:08:20.688–00:08:25.750 (Segment 183)
  - 00:08:58.438–00:09:14.750 (Segments 197–202)
  - 00:09:46.438–00:10:40.625 (Segments 213, 217, 228, 232)
  - 00:11:01.813–00:11:07.688 (Segments 244–245)
  - 00:12:44.688–00:13:02.688 (Segments 288–293)
  - 00:13:09.438–00:13:25.875 (Segments 297–301)
  - 00:13:43.250–00:13:56.438 (Segments 308, 310, 312)
  - 00:14:15.875–00:14:36.625 (Segments 319–325)

- SOL Orbital Laser (and related operations)
  - 01:30:06.063–01:30:28.000 (Segments 1769, 1772, 1775–1776)
  - 01:35:26.375–01:35:31.125 (Segment 1874)
  - 01:36:18.938–01:38:46.750 (Segments 1893, 1906–1907, 1922–1925, 1943)

- Stadium / Akira Containers / Awakening
  - 01:31:07.563–01:31:17.188 (Segments 1790–1792)
  - 01:31:45.625–01:32:03.375 (Segments 1802–1806)
  - 01:33:01.063–01:33:10.063 (Segments 1824–1825)
  - 01:34:11.500–01:34:32.313 (Segments 1843–1844, 1851–1852)
  - 01:38:10.188–01:38:12.875 (Segment 1930)
  - 01:41:50.250–01:45:15.125 (Segments 1995–2017, 2022–2054, 2061–2064)
  - 01:45:28.313–01:46:23.563 (Segments 2072, 2078, 2080–2085, 2088, 2092–2093)

- Final Confrontation / Metamorphosis
  - 01:46:55.938–01:47:03.750 (Segments 2106–2107)
  - 01:48:00.500–01:50:51.750 (Representative: 2124–2136, 2145–2150, 2159–2167, 2174–2180)
  - 01:51:21.813–01:51:51.500 (Segments 2190, 2196)
  - 01:56:53.313–02:00:04.125 (Late coda anchors: 2285, 2340)

Note: The anchors above are representative ranges to speed retrieval; neighboring segments often continue the same beat.
