#!/usr/bin/env python3
"""
TWR (Two-Way Ranging) Visualization Tool
Reads binary distance data from serial port and displays real-time graphs
"""

import sys
import struct
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from datetime import datetime
import numpy as np

# Configuration
SERIAL_PORT = None  # Auto-detect if None
BAUD_RATE = 115200
MAX_HISTORY = 100  # Number of data points to keep in history
UPDATE_INTERVAL = 50  # ms between plot updates

# Packet format constants
PACKET_HEADER = b'\xAA\x55'
MAX_ANCHORS = 8

class TwrParser:
    """Parser for TWR binary protocol"""
    
    def __init__(self):
        self.buffer = bytearray()
    
    def add_data(self, data):
        """Add incoming data to buffer"""
        self.buffer.extend(data)
    
    def parse_packet(self):
        """
        Try to parse a complete packet from buffer
        Returns: dict with anchor data or None if no complete packet
        
        Packet format:
        [0xAA 0x55] [length u16] [timestamp u32] [anchorCount u8]
        For each anchor: [anchorId u8] [distance float] [posValid u8] [x float] [y float] [z float]
        """
        # Look for packet header
        while len(self.buffer) >= 2:
            # Find header
            if self.buffer[0] == 0xAA and self.buffer[1] == 0x55:
                # Check if we have enough data for header + length
                if len(self.buffer) < 4:
                    return None
                
                # Read length
                length = struct.unpack('<H', self.buffer[2:4])[0]
                total_len = 4 + length  # header + length field + payload
                
                # Check if we have complete packet
                if len(self.buffer) < total_len:
                    return None
                
                # Extract packet
                packet_data = self.buffer[4:total_len]
                self.buffer = self.buffer[total_len:]  # Remove parsed packet
                
                # Parse payload
                try:
                    parsed = self._parse_payload(packet_data)
                    if parsed:
                        # Debug: print parsed packet info
                        print(f"Packet: ts={parsed['timestamp']}, anchors={len(parsed['anchors'])}")
                        for anc in parsed['anchors']:
                            print(f"  Anchor 0x{anc['id']:02X}: dist={anc['distance']:.3f}m")
                    return parsed
                except Exception as e:
                    print(f"Error parsing packet: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                # Skip invalid byte
                self.buffer = self.buffer[1:]
        
        return None
    
    def _parse_payload(self, data):
        """Parse packet payload"""
        if len(data) < 5:
            return None
        
        # Parse header
        timestamp, anchor_count = struct.unpack('<IB', data[0:5])
        
        anchors = []
        offset = 5
        
        for i in range(anchor_count):
            if offset + 18 > len(data):  # Each anchor is 18 bytes
                break
            
            # Parse anchor data: [id:1][dist:4][valid:1][x:4][y:4][z:4]
            anchor_id = data[offset]
            distance = struct.unpack('<f', data[offset+1:offset+5])[0]
            pos_valid = data[offset+5]
            x, y, z = struct.unpack('<fff', data[offset+6:offset+18])
            
            anchors.append({
                'id': anchor_id,
                'distance': distance,
                'position_valid': pos_valid == 1,
                'position': (x, y, z)
            })
            
            offset += 18
        
        return {
            'timestamp': timestamp,
            'anchors': anchors
        }


class TwrVisualizer:
    """Real-time TWR distance visualizer"""
    
    def __init__(self, port, baud_rate=115200):
        self.port = port
        self.baud_rate = baud_rate
        self.parser = TwrParser()
        self.serial = None
        
        # Data storage: {anchor_id: deque of (time, distance)}
        self.anchor_data = {}
        self.max_history = MAX_HISTORY
        
        # Matplotlib setup
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.lines = {}  # {anchor_id: line_object}
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # Start time for relative timestamps
        self.start_time = None
    
    def connect(self):
        """Connect to serial port"""
        try:
            self.serial = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            print(f"Connected to {self.port} at {self.baud_rate} baud")
            return True
        except Exception as e:
            print(f"Error connecting to {self.port}: {e}")
            return False
    
    def update_plot(self, frame):
        """Update plot with new data (called by animation)"""
        if self.serial is None or not self.serial.is_open:
            return list(self.lines.values())
        
        # Read available data
        try:
            while self.serial.in_waiting > 0:
                data = self.serial.read(self.serial.in_waiting)
                self.parser.add_data(data)
                
                # Parse all available packets
                while True:
                    packet = self.parser.parse_packet()
                    if packet is None:
                        break
                    self._process_packet(packet)
        except Exception as e:
            print(f"Error reading serial: {e}")
        
        # Update plot
        self._update_graph()
        
        return list(self.lines.values())
    
    def _process_packet(self, packet):
        """Process parsed packet and store data"""
        if self.start_time is None:
            self.start_time = datetime.now()
        
        # Calculate relative time in seconds
        rel_time = (datetime.now() - self.start_time).total_seconds()
        
        for anchor in packet['anchors']:
            anchor_id = anchor['id']
            distance = anchor['distance']
            
            # Skip NaN or invalid distances
            if not np.isfinite(distance) or distance < 0:
                continue
            
            # Initialize deque for new anchor
            if anchor_id not in self.anchor_data:
                self.anchor_data[anchor_id] = deque(maxlen=self.max_history)
            
            # Add data point
            print(f"Anchor 0x{anchor_id:02X}: {distance:.2f} m at {rel_time:.2f} s")
            self.anchor_data[anchor_id].append((rel_time, distance))
    
    def _update_graph(self):
        """Update graph with current data"""
        self.ax.clear()
        
        # Plot each anchor
        for anchor_id, data_points in sorted(self.anchor_data.items()):
            if len(data_points) == 0:
                continue
            
            times, distances = zip(*data_points)
            
            color_idx = anchor_id % len(self.colors)
            line, = self.ax.plot(times, distances, 
                                marker='o', 
                                markersize=3,
                                linewidth=1.5,
                                label=f'Anchor 0x{anchor_id:02X}',
                                color=self.colors[color_idx])
            
            self.lines[anchor_id] = line
        
        # Configure plot
        self.ax.set_xlabel('Time (seconds)', fontsize=12)
        self.ax.set_ylabel('Distance (meters)', fontsize=12)
        self.ax.set_title('TWR Distance Measurements - Real-time', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')
        
        # Auto-scale y-axis with some margin
        if self.anchor_data:
            all_distances = [d for data in self.anchor_data.values() for _, d in data]
            if all_distances:
                min_dist = min(all_distances)
                max_dist = max(all_distances)
                margin = (max_dist - min_dist) * 0.1 if max_dist > min_dist else 1.0
                self.ax.set_ylim(max(0, min_dist - margin), max_dist + margin)
    
    def run(self):
        """Start visualization"""
        if not self.connect():
            return
        
        print(f"Starting TWR visualization...")
        print(f"Recording up to {self.max_history} data points per anchor")
        print("Press Ctrl+C to stop\n")
        
        # Create animation
        ani = animation.FuncAnimation(
            self.fig,
            self.update_plot,
            interval=UPDATE_INTERVAL,
            blit=False,
            cache_frame_data=False
        )
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            if self.serial and self.serial.is_open:
                self.serial.close()
                print("Serial port closed")


def list_serial_ports():
    """List available serial ports"""
    ports = serial.tools.list_ports.comports()
    return [(p.device, p.description) for p in ports]


def select_serial_port():
    """Interactive serial port selection"""
    ports = list_serial_ports()
    
    if not ports:
        print("No serial ports found!")
        return None
    
    print("\nAvailable serial ports:")
    for i, (device, description) in enumerate(ports, 1):
        print(f"  {i}. {device} - {description}")
    
    while True:
        try:
            choice = input(f"\nSelect port (1-{len(ports)}) or press Enter for auto-detect: ").strip()
            if not choice:
                return ports[0][0]  # Return first port
            
            idx = int(choice) - 1
            if 0 <= idx < len(ports):
                return ports[idx][0]
            else:
                print(f"Invalid choice. Enter 1-{len(ports)}")
        except ValueError:
            print("Invalid input. Enter a number.")
        except KeyboardInterrupt:
            print("\nCancelled")
            return None


def main():
    """Main entry point"""
    print("=" * 60)
    print("TWR Distance Visualization Tool")
    print("=" * 60)
    
    # Select serial port
    if SERIAL_PORT is None:
        port = select_serial_port()
        if port is None:
            return 1
    else:
        port = SERIAL_PORT
    
    # Create and run visualizer
    visualizer = TwrVisualizer(port, BAUD_RATE)
    visualizer.run()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
