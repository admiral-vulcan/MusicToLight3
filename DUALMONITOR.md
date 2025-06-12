# Dual Monitor Setup with Pygame on Raspberry Pi (LXDE/X11)

This guide documents a working configuration for running a **single fullscreen Pygame window across two extended HDMI monitors** on a Raspberry Pi using LXDE. This approach allows rendering distinct content per screen from within one process.

### ✅ Confirmed Working With:

* **Pygame**: 2.6.1
* **SDL**: 2.28.4
* **Python**: 3.9.2
* **Display Manager**: LXDE with X11
* **Graphics Output**: HDMI-1 (left) and HDMI-2 (right), both at 1280x720

### 🧪 Quirks and Crucial Workarounds

1. **Do NOT use the exact height of 720 pixels!**

   * Using `HEIGHT = 720` results in failure (e.g., only one screen shows output).
   * Instead: `HEIGHT = 719` (or any other value ≠ native resolution).

2. **Set these environment variables before initializing Pygame**:

   ```python
   os.environ['DISPLAY'] = ':0'
   os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'
   os.environ['SDL_VIDEO_CENTERED'] = '0'
   os.environ['SDL_VIDEO_FULLSCREEN_HEAD'] = '0'  # try '1' if needed
   ```

3. **Pygame Window Creation**:
   Use `NOFRAME` to simulate fullscreen across extended displays:

   ```python
   pygame.init()
   WIDTH = 2560
   HEIGHT = 719  # One pixel less than native 720
   win = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)
   ```

4. **Dual Output Configuration**:
   Make sure the Pi is set up to extend (not mirror) the display:

   ```bash
   xrandr --output HDMI-1 --mode 1280x720 --primary
   xrandr --output HDMI-2 --mode 1280x720 --right-of HDMI-1
   ```

   This can be placed in a `set_hdmi_resolutions.sh` script triggered on LXDE autostart.

### 🖥️ Result:

One Pygame window spans both monitors. You can then draw left/right content depending on `x` position.

### 🧠 Notes:

* Make sure the correct user is logged in graphically (LXDE autologin may behave differently across users).
* If display seems mirrored or "stuck", try attaching mouse/keyboard to trigger proper desktop behavior.
* SDL fullscreen and window management has quirks in dual-screen mode under X11 – treat it like a hack zone.

---

Have fun!

*Maintained by Felix, 2025*
