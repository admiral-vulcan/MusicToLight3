#!/bin/bash

export DISPLAY=:0


# Modeline aus cvt (für beide HDMI-Ausgänge erzwingen)
MODELINE='1280x720_60.00'
MODEDEF='74.50 1280 1344 1472 1664 720 723 728 748 -hsync +vsync'

# HDMI-1
if xrandr | grep -q "HDMI-1 connected"; then
    xrandr --newmode $MODELINE $MODEDEF 2>/dev/null
    xrandr --addmode HDMI-1 $MODELINE 2>/dev/null
    xrandr --output HDMI-1 --mode $MODELINE --primary
fi

# HDMI-2
if xrandr | grep -q "HDMI-2 connected"; then
    xrandr --newmode $MODELINE $MODEDEF 2>/dev/null
    xrandr --addmode HDMI-2 $MODELINE 2>/dev/null
    xrandr --output HDMI-2 --mode $MODELINE --right-of HDMI-1
fi

pkill lxpanel
