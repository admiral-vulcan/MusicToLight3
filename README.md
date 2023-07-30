# MusicToLight3
MusicToLight3 is a project that converts audio input into visual output on various DMX devices, LED strip(s), and more. It's designed to run on a Raspberry Pi 4 with Raspbian OS 64-bit.

## Features
- Visualizes audio input on DMX devices and LED strips.
- Provides adjustable color transitions and mappings based on the audio input.
- Optimized to run on Raspberry Pi 4.

## Prerequisites
- Raspberry Pi 4 with Raspbian OS 64-bit.
- Python 3.8 or newer.
- The following Python libraries: PyAudio, collections, time, math, numpy, Aubio, rpi_ws281x, random, scipy.
- Open Lighting Architecture (OLA) for DMX communication.

The project has dependencies that are distributed under various licenses:
- PyAudio: MIT License
- numpy: BSD License
- aubio: GNU General Public License (GPL)
- rpi_ws281x: MIT License
- scipy: BSD License
- OLA: GNU Lesser General Public License (LGPL)

## Installation
1. Ensure that Python 3.8 or newer is installed.
2. Install the necessary Python libraries with pip:
```pip install pyaudio numpy aubio rpi_ws281x scipy```
3. Clone this repository to your local machine.
4. Set up your DMX devices and LED strips as needed.
5. Install and configure [OLA](https://www.openlighting.org/ola/).

## Usage
Run the main script to start visualizing audio input. You can adjust the specific behaviors of the visualization in the settings file.

## Contributing
Contributions are welcome! Please fork this repository and open a pull request with your changes.

## License
MusicToLight3  Copyright (C) 2023  Felix Rau. 
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
For more information, see the [LICENSE.md](LICENSE.md) file.

## Contact
For questions or comments, please open an issue on this repository.
