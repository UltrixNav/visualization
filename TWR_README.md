# TWR Visualization Tool

Real-time visualization of Two-Way Ranging (TWR) distance measurements.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Connect tag device to USB/serial port

3. Run visualization:
```bash
python twr_vis.py
```

4. Select the serial port when prompted (or the first port will be auto-selected)

## Features

- **Real-time plotting**: Distance measurements update live as data arrives
- **Multiple anchors**: Displays distances to all detected anchors simultaneously
- **Auto-scaling**: Y-axis automatically adjusts to data range
- **Color-coded**: Each anchor has a unique color for easy identification
- **History tracking**: Shows up to 100 most recent measurements per anchor

## Binary Protocol Format

The tool expects binary packets from the tag in the following format:

```
[0xAA 0x55]          Header (2 bytes)
[length]             Payload length u16 (2 bytes)
[timestamp]          milliseconds u32 (4 bytes)
[anchorCount]        Number of anchors u8 (1 byte)

For each anchor (repeat anchorCount times):
  [anchorId]         Anchor ID u8 (1 byte)
  [distance]         Distance in meters float (4 bytes)
  [positionValid]    Position validity flag u8 (1 byte)
  [x, y, z]          Position coordinates float×3 (12 bytes)
```

## Configuration

Edit the constants at the top of `twr_vis.py`:

- `SERIAL_PORT`: Set to specific port (e.g., 'COM3') or `None` for auto-detect
- `BAUD_RATE`: Serial baud rate (default: 115200)
- `MAX_HISTORY`: Number of data points to keep in history (default: 100)
- `UPDATE_INTERVAL`: Milliseconds between plot updates (default: 50)

## Tag Configuration

Ensure the tag firmware has binary output enabled:

In `tag/src/main.cpp`:
```cpp
constexpr bool kUartBinaryOutput = true;  // Enable binary output for TWR
```

## Troubleshooting

**No data appearing on graph:**
- Check that the tag device is connected and serial port is correct
- Verify binary output is enabled in tag firmware
- Ensure correct baud rate (115200)
- Check that tag is in TWR mode (not TDOA3)

**Serial port permission errors (Linux/Mac):**
```bash
sudo usermod -a -G dialout $USER
# Then log out and back in
```

**Invalid distances (NaN, negative, very large):**
- Check that anchors and tag are properly synchronized
- Verify antenna delay calibration in firmware
- Ensure line-of-sight between tag and anchors

## Example Output

The graph will show:
- X-axis: Time in seconds since start
- Y-axis: Distance in meters
- Legend: Each anchor labeled with its hex ID (e.g., "Anchor 0x01")
- Grid: For easier reading of values
