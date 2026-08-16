# Sign Language Recognition — Data Collection Modes

## Easy Rule for Deciding

Ask yourself two questions:

1. **Does the sign require hand movement to express it?** → If yes, use **Motion mode**.
2. **Do both hands need to remain in a fixed position together?** → If yes, use **Two-hand mode**.
3. If neither applies → use **Single-hand (default) mode**.

---

## 1. Single-hand Static (`python main.py collect`)

One hand, **static position**, with no movement.

### Examples

- ASL letters: A, B, C, D, E, F, I, L, O, S, U, V, W, Y (all letters that don't require movement)
- Fist, Open palm, Thumbs up, Peace (✌️), OK sign
- Counting: 1–9 (with one hand)
- Pointing, "Call me" (👌🤙 gesture)

**Start from here** — it's the easiest and is good for testing the entire pipeline.

---

## 2. Two-hand Static (`python main.py collect --two-hand`)

**Both hands are used together**, but there is no movement — the sign is defined by the relationship between the two hands.

### Examples

- ASL "Table" (both palms facing downward)
- ASL "Book" (both hands positioned like opening a book)
- ASL "Same"/"Friend" (both index fingers close together/hooked)
- Hands joined together (like a Namaste gesture)
- A roof/house-shaped gesture made with both hands

**Caution:** If the sign actually involves *movement* (such as real clapping), it should go to **Motion mode** instead of Two-hand Static. However, Motion mode currently supports only a single hand — see below.

---

## 3. Motion Signs (`python main.py collect_motion`)

One hand, but the **shape is defined by a path/movement** — the starting and ending positions are different.

### Examples

- ASL letter **J** (drawing a hook-like motion)
- ASL letter **Z** (drawing a zigzag)
- Swipe left/right/up/down
- "Bye-bye" wave
- "Come here" (beckoning with the finger)
- ASL "Thank you" (moving the hand forward from the chin)

### ⚠️ Important Limitation

Motion mode currently supports only **single-hand** gestures. Two-hand motion signs, such as actual clapping, are not supported yet. This is mentioned as a **future extension** in the README.

---

## My Recommended Starting Order

1. First, test the complete pipeline (`collect` → `train` → `live`) with 3–4 easy signs in **Single-hand Static**, such as:
   - `fist`
   - `peace`
   - `thumbs_up`
   - `open_palm`

2. Once that works, try 1–2 gestures in **Motion mode**, such as:
   - `swipe_left`
   - `swipe_right`

3. Finally, move to **Two-hand mode** — this is the most complex, so it's better to do it last.

---

## Dataset and Model Separation

The datasets/models for the three modes are stored in completely separate files:

- `samples.csv` — Single-hand static
- `samples_2h.csv` — Two-hand static
- `samples_motion.csv` — Motion signs