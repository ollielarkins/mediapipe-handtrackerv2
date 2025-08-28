import os, subprocess, yt_dlp, cv2, tempfile, time, sys, csv, shutil
from datetime import timedelta
import mediapipe as mp
from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.table import Table
from rich.progress import Progress
from rich.text import Text
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import math

# --- Setup paths ---
# Get script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create folders relative to script location
video_folder = os.path.join(script_dir, "videos")
tracked_folder = os.path.join(script_dir, "tracked") 
csv_folder = os.path.join(script_dir, "csv_data")
reports_folder = os.path.join(script_dir, "reports")
cache_file = os.path.join(script_dir, "url_cache.txt")

# Create required folders
for folder in [video_folder, tracked_folder, csv_folder, reports_folder]:
    os.makedirs(folder, exist_ok=True)

# Find FFmpeg executables in PATH
ffmpeg_path = shutil.which("ffmpeg") or "ffmpeg.exe"
ffplay_path = shutil.which("ffplay") or "ffplay.exe"

# --- Rich console ---
console = Console()
ascii_title = r"""
                 _ _         ___ _                                    _ _____                _    _             
  /\/\   ___  __| (_) __ _  / _ (_)_ __   ___    /\  /\__ _ _ __   __| /__   \_ __ __ _  ___| | _(_)_ __   __ _ 
 /    \ / _ \/ _` | |/ _` |/ /_)/ | '_ \ / _ \  / /_/ / _` | '_ \ / _` | / /\/ '__/ _` |/ __| |/ / | '_ \ / _` |
/ /\/\ \  __/ (_| | | (_| / ___/| | |_) |  __/ / __  / (_| | | | | (_| |/ /  | | | (_| | (__|   <| | | | | (_| |
\/    \/\___|\__,_|_|\__,_\/    |_| .__/ \___| \/ /_/ \__,_|_| |_|\__,_|\/   |_|  \__,_|\___|_|\_\_|_| |_|\__, |
                                  |_|                                                                     |___/ 
"""
for i in range(6):
    console.clear()
    border = "bold bright_blue" if i % 2 == 0 else "bold white"
    console.print(Panel(Align.center(ascii_title, vertical="middle"), border_style=border, expand=True))
    time.sleep(0.5)
console.clear()

# --- URL Cache ---
url_cache = {}
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                url, filename = line.split("|||")
                url_cache[url] = filename

# --- Helper functions ---
def list_existing_videos():
    return [f for f in os.listdir(video_folder) if f.lower().endswith(".mp4")]

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Unable to open"
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps else 0
    cap.release()
    return f"{str(timedelta(seconds=int(duration_sec)))} | {width}x{height} | {fps:.2f} FPS"

def clean_old_csv_files(base_name):
    """Delete all existing CSV files for the same video"""
    deleted_count = 0
    csv_files_found = []
    
    # Find all CSV files that match this video
    for f in os.listdir(csv_folder):
        if f.startswith(base_name) and f.endswith("_hand_data.csv"):
            csv_files_found.append(f)
    
    # Delete found CSV files
    for csv_file in csv_files_found:
        try:
            os.remove(os.path.join(csv_folder, csv_file))
            deleted_count += 1
            console.print(f"[yellow]Deleted old CSV:[/yellow] {csv_file}")
        except Exception as e:
            console.print(f"[red]Could not delete CSV {csv_file}: {e}[/red]")
    
    if deleted_count > 0:
        console.print(f"[green]Cleaned up {deleted_count} old CSV file(s) for {base_name}[/green]")
    elif csv_files_found:
        console.print(f"[red]Found {len(csv_files_found)} old CSV file(s) but could not delete them[/red]")
    
    return deleted_count

def clean_old_report_files(base_name):
    """Delete all existing report files for the same video"""
    deleted_count = 0
    report_files_found = []
    
    # File patterns to look for
    patterns = [
        f"{base_name}_3d_trajectory.html",
        f"{base_name}_tracking_report.txt"
    ]
    
    # Find all report files that match this video
    for f in os.listdir(reports_folder):
        for pattern in patterns:
            if f == pattern:
                report_files_found.append(f)
                break
    
    # Delete found report files
    for report_file in report_files_found:
        try:
            os.remove(os.path.join(reports_folder, report_file))
            deleted_count += 1
            console.print(f"[yellow]Deleted old report:[/yellow] {report_file}")
        except Exception as e:
            console.print(f"[red]Could not delete report {report_file}: {e}[/red]")
    
    if deleted_count > 0:
        console.print(f"[green]Cleaned up {deleted_count} old report file(s) for {base_name}[/green]")
    elif report_files_found:
        console.print(f"[red]Found {len(report_files_found)} old report file(s) but could not delete them[/red]")
    
    return deleted_count

def create_ascii_heatmap(csv_data, width, height, base_name):
    """Create enhanced ASCII heatmap for CLI display"""
    console.print("\n" + "="*80)
    console.print(Panel(
        Align.center(f"[bold cyan]HAND MOVEMENT HEATMAP[/bold cyan]\n[dim]{base_name}[/dim]", 
                     vertical="middle"), 
        border_style="bold cyan", 
        expand=True
    ))
    
    if not csv_data:
        console.print("[red]No tracking data available for heatmap[/red]")
        return
    
    # Create 2D grid for heatmap (increased resolution for better clarity)
    grid_width, grid_height = 80, 25
    heatmap = np.zeros((grid_height, grid_width))
    
    # Separate data by hand for better analysis
    left_data = [d for d in csv_data if d['hand'] == 'Left']
    right_data = [d for d in csv_data if d['hand'] == 'Right']
    
    # Process tracking data
    for entry in csv_data:
        # Convert normalized coordinates to grid positions
        grid_x = int(entry['wrist_x'] * (grid_width - 1))
        grid_y = int(entry['wrist_y'] * (grid_height - 1))
        
        # Ensure within bounds
        grid_x = max(0, min(grid_width - 1, grid_x))
        grid_y = max(0, min(grid_height - 1, grid_y))
        
        # Weight by hand type for better visualization
        weight = 1.5 if entry['hand'] == 'Right' else 1.0
        heatmap[grid_y][grid_x] += weight
    
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Enhanced ASCII visualization with better contrast
    chars = [' ', '·', '░', '▒', '▓', '█']
    colors = ['black', 'dim blue', 'blue', 'cyan', 'yellow', 'red']
    
    # Add border
    console.print("┌" + "─" * grid_width + "┐")
    
    for i, row in enumerate(heatmap):
        line_chars = []
        for val in row:
            char_idx = int(val * (len(chars) - 1))
            char = chars[char_idx]
            color = colors[char_idx]
            line_chars.append(f"[{color}]{char}[/{color}]")
        
        # Add row indicators for orientation
        if i == 0:
            console.print(f"│{''.join(line_chars)}│ [dim]← Top[/dim]")
        elif i == len(heatmap) - 1:
            console.print(f"│{''.join(line_chars)}│ [dim]← Bottom[/dim]")
        elif i == len(heatmap) // 2:
            console.print(f"│{''.join(line_chars)}│ [dim]← Center[/dim]")
        else:
            console.print(f"│{''.join(line_chars)}│")
    
    console.print("└" + "─" * grid_width + "┘")
    
    # Enhanced legend and statistics
    legend_table = Table(show_header=False, box=None, padding=(0, 1))
    legend_table.add_column("Symbol", style="bold")
    legend_table.add_column("Intensity", style="dim")
    legend_table.add_column("Color", style="bold")
    
    legend_items = [
        (" ", "No Activity", "black"),
        ("·", "Minimal", "dim blue"), 
        ("░", "Low", "blue"),
        ("▒", "Medium", "cyan"),
        ("▓", "High", "yellow"),
        ("█", "Very High", "red")
    ]
    
    for symbol, intensity, color in legend_items:
        legend_table.add_row(
            f"[{color}]{symbol}[/{color}]",
            intensity,
            color.title()
        )
    
    console.print("\n[bold]Legend:[/bold]")
    console.print(legend_table)
    
    # Statistics summary
    stats_table = Table(title="Movement Statistics", show_header=True, header_style="bold magenta")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")
    
    total_detections = len(csv_data)
    left_detections = len(left_data)
    right_detections = len(right_data)
    coverage = np.count_nonzero(heatmap) / heatmap.size * 100
    
    stats_table.add_row("Total Hand Detections", str(total_detections))
    stats_table.add_row("Left Hand Detections", f"{left_detections} ({left_detections/total_detections*100:.1f}%)" if total_detections > 0 else "0")
    stats_table.add_row("Right Hand Detections", f"{right_detections} ({right_detections/total_detections*100:.1f}%)" if total_detections > 0 else "0")
    stats_table.add_row("Screen Coverage", f"{coverage:.1f}%")
    stats_table.add_row("Grid Resolution", f"{width} × {height}")
    stats_table.add_row("Video Resolution", f"{width} × {height}")
    
    console.print("\n")
    console.print(stats_table)
    console.print("="*80)

def create_3d_trajectory(csv_data, base_name):
    """Create user-friendly 3D trajectory visualization using Plotly"""
    if not csv_data:
        console.print("[red]No tracking data available for 3D trajectory[/red]")
        return
    
    console.print("[green]Generating 3D trajectory visualization...[/green]")
    
    # Separate data by hand
    left_data = [d for d in csv_data if d['hand'] == 'Left']
    right_data = [d for d in csv_data if d['hand'] == 'Right']
    
    fig = go.Figure()
    
    # Add left hand trajectory
    if left_data:
        left_data.sort(key=lambda x: x['frame'])  # Sort by frame
        x_left = [d['wrist_x'] for d in left_data]
        y_left = [d['wrist_y'] for d in left_data]
        z_left = [d['wrist_z'] for d in left_data]
        frames_left = [d['frame'] for d in left_data]
        
        fig.add_trace(go.Scatter3d(
            x=x_left, y=y_left, z=z_left,
            mode='lines+markers',
            marker=dict(size=4, color=frames_left, colorscale='Reds', 
                       showscale=True, colorbar=dict(title="Video Frame", x=1.15)),
            line=dict(color='red', width=6),
            name='Left Hand Path',
            hovertemplate='<b>Left Hand</b><br>' +
                         'Horizontal: %{x:.3f}<br>' +
                         'Vertical: %{y:.3f}<br>' +
                         'Depth: %{z:.3f}<br>' +
                         'Frame: %{marker.color}<br>' +
                         '<i>Red line shows left hand movement</i><extra></extra>'
        ))
    
    # Add right hand trajectory
    if right_data:
        right_data.sort(key=lambda x: x['frame'])  # Sort by frame
        x_right = [d['wrist_x'] for d in right_data]
        y_right = [d['wrist_y'] for d in right_data]
        z_right = [d['wrist_z'] for d in right_data]
        frames_right = [d['frame'] for d in right_data]
        
        fig.add_trace(go.Scatter3d(
            x=x_right, y=y_right, z=z_right,
            mode='lines+markers',
            marker=dict(size=4, color=frames_right, colorscale='Blues', 
                       showscale=True, colorbar=dict(title="Video Frame", x=1.25)),
            line=dict(color='blue', width=6),
            name='Right Hand Path',
            hovertemplate='<b>Right Hand</b><br>' +
                         'Horizontal: %{x:.3f}<br>' +
                         'Vertical: %{y:.3f}<br>' +
                         'Depth: %{z:.3f}<br>' +
                         'Frame: %{marker.color}<br>' +
                         '<i>Blue line shows right hand movement</i><extra></extra>'
        ))
    
    # Add starting points for clarity
    if left_data:
        fig.add_trace(go.Scatter3d(
            x=[x_left[0]], y=[y_left[0]], z=[z_left[0]],
            mode='markers',
            marker=dict(size=12, color='darkred', symbol='diamond'),
            name='Left Hand Start',
            hovertemplate='<b>Left Hand Starting Point</b><br>Frame: %{text}<extra></extra>',
            text=[frames_left[0]]
        ))
    
    if right_data:
        fig.add_trace(go.Scatter3d(
            x=[x_right[0]], y=[y_right[0]], z=[z_right[0]],
            mode='markers',
            marker=dict(size=12, color='darkblue', symbol='diamond'),
            name='Right Hand Start',
            hovertemplate='<b>Right Hand Starting Point</b><br>Frame: %{text}<extra></extra>',
            text=[frames_right[0]]
        ))
    
    # Create comprehensive title and annotations
    total_frames = max([d['frame'] for d in csv_data]) if csv_data else 0
    left_count = len(left_data)
    right_count = len(right_data)
    
    title_text = f"<b>Hand Movement Analysis: {base_name}</b><br>" + \
                f"<span style='font-size: 14px;'>Left Hand: {left_count} detections | " + \
                f"Right Hand: {right_count} detections | Total Frames: {total_frames}</span>"
    
    # Customize layout with detailed explanations
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title='<b>Horizontal Position</b><br><i>(0 = Left Edge, 1 = Right Edge)</i>',
            yaxis_title='<b>Vertical Position</b><br><i>(0 = Top Edge, 1 = Bottom Edge)</i>',
            zaxis_title='<b>Depth Position</b><br><i>(Closer to camera = Larger values)</i>',
            bgcolor='rgba(240,240,240,0.9)',
            xaxis=dict(gridcolor='gray', range=[0, 1], dtick=0.2),
            yaxis=dict(gridcolor='gray', range=[0, 1], dtick=0.2),
            zaxis=dict(gridcolor='gray'),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),  # Better default viewing angle
                up=dict(x=0, y=0, z=1)
            )
        ),
        font=dict(size=12),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        annotations=[
            dict(
                text="<b>How to Read This Graph:</b><br>" +
                     "• Red lines = Left hand movement path<br>" +
                     "• Blue lines = Right hand movement path<br>" +
                     "• Diamonds = Starting positions<br>" +
                     "• Darker colors = Earlier in video<br>" +
                     "• Lighter colors = Later in video<br>" +
                     "• Use mouse to rotate and zoom",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.02, y=0.02,
                xanchor="left", yanchor="bottom",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=11)
            )
        ]
    )
    
    # Save and show
    trajectory_file = os.path.join(reports_folder, f"{base_name}_3d_trajectory.html")
    fig.write_html(trajectory_file)
    console.print(f"[green]3D trajectory saved:[/green] {trajectory_file}")
    
    # Auto-open in browser with user-friendly message
    try:
        import webbrowser
        webbrowser.open(f'file://{os.path.abspath(trajectory_file)}')
        console.print("[cyan]Opening interactive 3D hand movement visualization...[/cyan]")
        console.print("[dim]Tip: Use your mouse to rotate, zoom, and explore the 3D space![/dim]")
    except:
        console.print("[yellow]Could not auto-open browser. Please open the HTML file manually.[/yellow]")

def calculate_movement_stats(csv_data, fps):
    """Calculate detailed movement statistics"""
    if not csv_data:
        return {}
    
    stats = {'left': {}, 'right': {}, 'combined': {}}
    
    for hand_type in ['Left', 'Right']:
        hand_data = [d for d in csv_data if d['hand'] == hand_type]
        if not hand_data:
            continue
        
        # Sort by frame number to ensure correct order
        hand_data.sort(key=lambda x: x['frame'])
        
        hand_key = hand_type.lower()
        
        # Basic counts
        stats[hand_key]['total_detections'] = len(hand_data)
        stats[hand_key]['frames_active'] = len(set(d['frame'] for d in hand_data))
        
        # Calculate distances and speeds
        distances = []
        speeds = []
        
        for i in range(1, len(hand_data)):
            prev = hand_data[i-1]
            curr = hand_data[i]
            
            # 3D distance between consecutive detections
            dx = curr['wrist_x'] - prev['wrist_x']
            dy = curr['wrist_y'] - prev['wrist_y']
            dz = curr['wrist_z'] - prev['wrist_z']
            
            distance = math.sqrt(dx**2 + dy**2 + dz**2)
            distances.append(distance)
            
            # Speed calculation (units per second)
            frame_diff = curr['frame'] - prev['frame']
            if frame_diff > 0 and fps > 0:
                time_diff = frame_diff / fps  # Convert frame difference to time
                speed = distance / time_diff
                speeds.append(speed)
        
        # Store distance statistics
        if distances:
            stats[hand_key]['total_distance'] = sum(distances)
            stats[hand_key]['avg_distance_per_frame'] = sum(distances) / len(distances)
            stats[hand_key]['max_distance_per_frame'] = max(distances)
            stats[hand_key]['min_distance_per_frame'] = min(distances)
        else:
            stats[hand_key]['total_distance'] = 0
            stats[hand_key]['avg_distance_per_frame'] = 0
            stats[hand_key]['max_distance_per_frame'] = 0
            stats[hand_key]['min_distance_per_frame'] = 0
        
        # Store speed statistics
        if speeds:
            stats[hand_key]['avg_speed'] = sum(speeds) / len(speeds)
            stats[hand_key]['max_speed'] = max(speeds)
            stats[hand_key]['min_speed'] = min(speeds)
            stats[hand_key]['speed_std'] = np.std(speeds) if len(speeds) > 1 else 0
        else:
            stats[hand_key]['avg_speed'] = 0
            stats[hand_key]['max_speed'] = 0
            stats[hand_key]['min_speed'] = 0
            stats[hand_key]['speed_std'] = 0
        
        # Position statistics
        x_positions = [d['wrist_x'] for d in hand_data]
        y_positions = [d['wrist_y'] for d in hand_data]
        z_positions = [d['wrist_z'] for d in hand_data]
        
        stats[hand_key]['position_ranges'] = {
            'x_range': max(x_positions) - min(x_positions) if x_positions else 0,
            'y_range': max(y_positions) - min(y_positions) if y_positions else 0,
            'z_range': max(z_positions) - min(z_positions) if z_positions else 0,
        }
        
        stats[hand_key]['center_of_mass'] = {
            'x': sum(x_positions) / len(x_positions) if x_positions else 0,
            'y': sum(y_positions) / len(y_positions) if y_positions else 0,
            'z': sum(z_positions) / len(z_positions) if z_positions else 0
        }
    
    # Combined statistics
    all_data = csv_data
    if all_data:
        stats['combined']['total_detections'] = len(all_data)
        stats['combined']['unique_frames'] = len(set(d['frame'] for d in all_data))
        max_frame = max(d['frame'] for d in all_data)
        stats['combined']['detection_rate'] = (stats['combined']['unique_frames'] / max_frame * 100) if max_frame > 0 else 0
    
    return stats

def generate_tracking_report(csv_data, base_name, fps, duration_sec):
    """Generate comprehensive tracking analysis report in CLI"""
    console.print("\n[bold cyan]═══ TRACKING ANALYSIS REPORT ═══[/bold cyan]")
    
    if not csv_data:
        console.print("[red]No tracking data available for report[/red]")
        return
    
    # Calculate statistics
    stats = calculate_movement_stats(csv_data, fps)
    
    # Video Information Section
    report_table = Table(title="Video Information", show_header=True, header_style="bold magenta")
    report_table.add_column("Property", style="cyan", width=20)
    report_table.add_column("Value", style="white")
    
    report_table.add_row("Video Name", base_name)
    report_table.add_row("Duration", f"{duration_sec:.2f} seconds")
    report_table.add_row("FPS", f"{fps:.2f}")
    report_table.add_row("Total Frames", f"{int(duration_sec * fps)}")
    
    console.print(report_table)
    
    # Detection Summary
    detection_table = Table(title="Hand Detection Summary", show_header=True, header_style="bold magenta")
    detection_table.add_column("Hand", style="cyan", width=15)
    detection_table.add_column("Detections", style="green", width=12)
    detection_table.add_column("Active Frames", style="yellow", width=15)
    detection_table.add_column("Detection Rate", style="blue", width=15)
    
    total_frames = int(duration_sec * fps)
    
    for hand in ['left', 'right']:
        if hand in stats and 'total_detections' in stats[hand]:
            detections = stats[hand]['total_detections']
            active_frames = stats[hand]['frames_active']
            detection_rate = (active_frames / total_frames * 100) if total_frames > 0 else 0
            
            detection_table.add_row(
                hand.title(),
                str(detections),
                str(active_frames),
                f"{detection_rate:.1f}%"
            )
    
    console.print(detection_table)
    
    # Movement Analysis
    movement_table = Table(title="Movement Analysis", show_header=True, header_style="bold magenta")
    movement_table.add_column("Metric", style="cyan", width=25)
    movement_table.add_column("Left Hand", style="red", width=15)
    movement_table.add_column("Right Hand", style="blue", width=15)
    
    metrics = [
        ("Total Distance", "total_distance", ":.4f"),
        ("Avg Speed (units/sec)", "avg_speed", ":.4f"),
        ("Max Speed (units/sec)", "max_speed", ":.4f"),
        ("Speed Variation (std)", "speed_std", ":.4f"),
        ("X Movement Range", "position_ranges.x_range", ":.3f"),
        ("Y Movement Range", "position_ranges.y_range", ":.3f"),
        ("Z Movement Range", "position_ranges.z_range", ":.3f"),
    ]
    
    for metric_name, metric_key, format_str in metrics:
        left_val = "N/A"
        right_val = "N/A"
        
        # Handle nested keys
        if 'left' in stats:
            try:
                if '.' in metric_key:
                    keys = metric_key.split('.')
                    val = stats['left']
                    for k in keys:
                        val = val[k]
                    left_val = format(val, format_str)
                elif metric_key in stats['left']:
                    left_val = format(stats['left'][metric_key], format_str)
            except:
                pass
        
        if 'right' in stats:
            try:
                if '.' in metric_key:
                    keys = metric_key.split('.')
                    val = stats['right']
                    for k in keys:
                        val = val[k]
                    right_val = format(val, format_str)
                elif metric_key in stats['right']:
                    right_val = format(stats['right'][metric_key], format_str)
            except:
                pass
        
        movement_table.add_row(metric_name, left_val, right_val)
    
    console.print(movement_table)
    
    # Save detailed report to file
    report_file = os.path.join(reports_folder, f"{base_name}_tracking_report.txt")
    with open(report_file, 'w') as f:
        f.write(f"HAND TRACKING ANALYSIS REPORT\n")
        f.write(f"{'='*50}\n")
        f.write(f"Video: {base_name}\n")
        f.write(f"Duration: {duration_sec:.2f} seconds\n")
        f.write(f"FPS: {fps:.2f}\n")
        f.write(f"Total Frames: {int(duration_sec * fps)}\n\n")
        
        f.write("DETECTION SUMMARY:\n")
        f.write("-" * 30 + "\n")
        for hand in ['left', 'right']:
            if hand in stats and 'total_detections' in stats[hand]:
                f.write(f"{hand.title()} Hand:\n")
                f.write(f"  - Total Detections: {stats[hand]['total_detections']}\n")
                f.write(f"  - Active Frames: {stats[hand]['frames_active']}\n")
                f.write(f"  - Detection Rate: {stats[hand]['frames_active']/total_frames*100:.1f}%\n")
        
        f.write("\nMOVEMENT ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        for hand in ['left', 'right']:
            if hand in stats and 'total_distance' in stats[hand]:
                f.write(f"{hand.title()} Hand Movement:\n")
                for key, value in stats[hand].items():
                    if isinstance(value, dict):
                        f.write(f"  - {key}:\n")
                        for k, v in value.items():
                            f.write(f"    - {k}: {v:.4f}\n")
                    elif isinstance(value, (int, float)):
                        f.write(f"  - {key}: {value:.4f}\n")
                    else:
                        f.write(f"  - {key}: {value}\n")
                f.write("\n")
    
    console.print(f"\n[green]Detailed report saved:[/green] {report_file}")

# Silence Mediapipe/TensorFlow warnings
sys.stderr = open(os.devnull, 'w')
_ = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
time.sleep(0.3)
_.close()
sys.stderr = sys.__stderr__

# --- Main Menu Loop ---
while True:
    console.clear()
    table = Table(title="", show_lines=True, expand=True)
    table.add_column("No.", justify="center", style="cyan")
    table.add_column("Filename", justify="left", style="magenta")
    table.add_column("Info", justify="left", style="green")

    existing_videos = list_existing_videos()
    if existing_videos:
        for i, v in enumerate(existing_videos, 1):
            info = get_video_info(os.path.join(video_folder, v))
            table.add_row(str(i), v, info)
    else:
        table.add_row("-", "[red]None[/red]", "-")

    console.print(table)
    choice = console.input("\nEnter URL, filename/number, 'delete #', 'clear cache', or 'exit': ").strip()
    if choice.lower() == "exit":
        sys.exit(0)

    try:
        if choice.lower() == "clear cache":
            if os.path.exists(cache_file):
                os.remove(cache_file)
                url_cache.clear()
                console.print("[green]URL cache cleared.[/green]")
            continue
        if choice.lower().startswith("delete"):
            try:
                idx = int(choice.split()[1]) - 1
                vid_to_delete = existing_videos[idx]
                os.remove(os.path.join(video_folder, vid_to_delete))
                console.print(f"[yellow]Deleted:[/yellow] {vid_to_delete}")
            except:
                console.print("[red]Invalid delete command.[/red]")
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(existing_videos):
            output_path = os.path.join(video_folder, existing_videos[int(choice)-1])
        elif choice in existing_videos:
            output_path = os.path.join(video_folder, choice)
        elif choice in url_cache:
            output_path = os.path.join(video_folder, url_cache[choice])
        else:
            url = choice
            video_name = console.input("Enter a name for this video (no extension): ").strip()
            output_name = f"{video_name}.mp4"
            output_path = os.path.join(video_folder, output_name)

            if not os.path.exists(output_path):
                ydl_opts = {
                    "format": "bestvideo+bestaudio/best",
                    "outtmpl": os.path.join(video_folder, "tmp.%(ext)s"),
                    "merge_output_format": "mp4",
                    "ffmpeg_location": ffmpeg_path,
                    "quiet": True,
                    "no_warnings": True,
                }
                console.print("[green]Downloading video...[/green]")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url)
                    downloaded_file = os.path.join(video_folder, "tmp." + info.get("ext", "mp4"))

                subprocess.run([ffmpeg_path, "-i", downloaded_file, "-vf", "scale=iw/2:ih/2",
                                "-c:v", "libx264", "-c:a", "aac", output_path],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(downloaded_file):
                    os.remove(downloaded_file)

                url_cache[url] = output_name
                with open(cache_file, "a") as f:
                    f.write(f"{url}|||{output_name}\n")
            else:
                console.print(f"[yellow]File already exists:[/yellow] {output_path}")

        if os.path.exists(output_path):
            break
        else:
            console.print("[red]Video not found. Try again.[/red]")
            continue

    except Exception as e:
        console.print(f"[red]Error: {e}. Try again.[/red]")
        continue

# --- CSV Setup & delete old files ---
base_name = os.path.splitext(os.path.basename(output_path))[0]
console.print(f"\n[cyan]Preparing to process:[/cyan] {base_name}")

# Clean up old CSV and report files for this video
clean_old_csv_files(base_name)
clean_old_report_files(base_name)

# --- Hand tracking ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=4,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(output_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps

temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
os.close(temp_fd)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

csv_file = os.path.join(csv_folder, f"{base_name}_hand_data.csv")
csv_columns = ["frame", "hand", "wrist_x", "wrist_y", "wrist_z", "num_landmarks"]
csv_data = []

console.print("[bold green]Processing video with hand tracking...[/bold green]")
with Progress() as progress:
    task = progress.add_task("[cyan]Tracking hands...", total=total_frames)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                color = (0,0,255) if handedness=="Left" else (255,0,0)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                                          mp_drawing.DrawingSpec(color=color, thickness=2))
                wrist = hand_landmarks.landmark[0]
                csv_data.append({
                    "frame": frame_idx,
                    "hand": handedness,
                    "wrist_x": wrist.x,
                    "wrist_y": wrist.y,
                    "wrist_z": wrist.z,
                    "num_landmarks": len(hand_landmarks.landmark)
                })

        out.write(frame)
        progress.update(task, advance=1)

cap.release()
out.release()
hands.close()
console.print("[bold green]Hand tracking complete![/bold green]")

# --- Write CSV ---
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()
    writer.writerows(csv_data)
console.print(f"[green]CSV saved:[/green] {csv_file}")

# --- Generate Analysis Features ---
console.print("\n[bold cyan]Generating analysis features...[/bold cyan]")

# 1. ASCII Heatmap (CLI display)
create_ascii_heatmap(csv_data, width, height, base_name)

# 2. 3D Trajectory Visualization (HTML file)
create_3d_trajectory(csv_data, base_name)

# 3. Tracking Report (CLI display + text file)
generate_tracking_report(csv_data, base_name, fps, duration_sec)

# --- Merge audio back ---
tracked_path = os.path.join(tracked_folder, f"tracked_{os.path.basename(output_path)}")
subprocess.run([ffmpeg_path, "-y", "-i", temp_path, "-i", output_path,
                "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", tracked_path],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os.remove(temp_path)

# --- Ask user for display preference ---
console.print("\n[bold cyan]Video processing complete![/bold cyan]")
while True:
    display_choice = console.input("\nHow would you like to view the result?\n[1] Side-by-side (original + tracked)\n[2] Tracked video only\n[3] Skip video playback\nEnter choice (1-3): ").strip()
    
    if display_choice == "1":
        # Play side-by-side with calculated positions
        console.print("[green]Playing videos side-by-side...[/green]")
        
        # Calculate window positions for side-by-side display
        # Get screen dimensions (approximate - adjust if needed)
        screen_width = 1920  # You can modify this based on your screen
        
        # Calculate positions to center both videos
        total_width = width * 2
        start_x = max(0, (screen_width - total_width) // 2)
        
        # Left window (original)
        left_x = start_x
        left_y = 100  # Some padding from top
        
        # Right window (tracked)
        right_x = start_x + width
        right_y = 100
        
        p1 = subprocess.Popen([ffplay_path, "-autoexit", "-window_title", f"Original - {os.path.basename(output_path)}",
                               "-x", str(width), "-y", str(height),
                               "-left", str(left_x), "-top", str(left_y),
                               output_path])
        time.sleep(0.25)
        p2 = subprocess.Popen([ffplay_path, "-autoexit", "-window_title", f"Tracked - {os.path.basename(tracked_path)}",
                               "-x", str(width), "-y", str(height),
                               "-left", str(right_x), "-top", str(right_y),
                               "-an", tracked_path])
        p1.wait()
        p2.wait()
        break
    elif display_choice == "2":
        # Play tracked video only
        console.print("[green]Playing tracked video...[/green]")
        p = subprocess.Popen([ffplay_path, "-autoexit", "-window_title", f"Tracked - {os.path.basename(tracked_path)}",
                              "-x", str(width), "-y", str(height), tracked_path])
        p.wait()
        break
    elif display_choice == "3":
        # Skip video playback
        console.print("[yellow]Skipping video playback.[/yellow]")
        break
    else:
        console.print("[red]Invalid choice. Please enter 1, 2, or 3.[/red]")

# --- Delete tracked video ---
try:
    os.remove(tracked_path)
    console.print(f"[green]Deleted tracked file:[/green] {tracked_path}")
except PermissionError:
    console.print(f"[red]Could not delete:[/red] {tracked_path}")
except FileNotFoundError:
    console.print(f"[yellow]Tracked file already removed:[/yellow] {tracked_path}")

console.print("\n[bold green]Process completed successfully![/bold green]")
console.print(f"[cyan]Files generated:[/cyan]")
console.print(f"  • CSV Data: {csv_file}")
console.print(f"  • 3D Trajectory: {os.path.join(reports_folder, f'{base_name}_3d_trajectory.html')}")
console.print(f"  • Analysis Report: {os.path.join(reports_folder, f'{base_name}_tracking_report.txt')}")