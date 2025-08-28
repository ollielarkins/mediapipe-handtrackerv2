# mediapipe-handtrackerv2
# MediaPipe Hand Tracking Tool 🤲

An advanced hand tracking analysis tool that processes videos to detect, track, and analyze hand movements using Google's MediaPipe framework. Features real-time hand detection, 3D trajectory visualization, movement analytics, and comprehensive reporting.


https://github.com/user-attachments/assets/79457fdb-9900-4082-a67b-96517a4f7a43


## 📷 Images (@oliver.larkins on Instagram)
<img width="1080" height="1350" alt="4" src="https://github.com/user-attachments/assets/276146a4-ee11-482d-bb60-8707cd53a283" />
<img width="1080" height="1350" alt="3" src="https://github.com/user-attachments/assets/1a7b7974-2690-4e03-a4e9-7ebc39869bf3" />
<img width="1080" height="1350" alt="2" src="https://github.com/user-attachments/assets/492587f3-b1a5-497d-9980-cee42f799007" />
<img width="1080" height="1350" alt="1" src="https://github.com/user-attachments/assets/e762a16c-6c56-4f92-84bd-4869e7b2e0e4" />


## ✨ Features

### Core Functionality
- **Multi-Hand Detection**: Track up to 4 hands simultaneously (left/right hand identification)
- **YouTube Integration**: Download and process videos directly from YouTube URLs
- **Real-time Processing**: Live hand tracking with visual overlay on video
- **3D Trajectory Analysis**: Interactive 3D visualization of hand movement paths
- **Movement Analytics**: Comprehensive statistical analysis of hand movements

### Analysis & Visualization
- **ASCII Heatmaps**: CLI-based movement heatmaps for quick analysis
- **Interactive 3D Plots**: Browser-based 3D trajectory visualization using Plotly
- **Movement Statistics**: Speed, distance, acceleration, and position analytics
- **Comprehensive Reports**: Detailed text and visual reports

### User Experience
- **Rich CLI Interface**: Beautiful command-line interface with progress bars and tables
- **Side-by-Side Playback**: Compare original and tracked videos simultaneously
- **Automatic Cleanup**: Smart file management with automatic cleanup of temporary files
- **URL Caching**: Remember downloaded videos to avoid re-downloading

## 🛠️ Installation

### Prerequisites
- Python 3.7+ (Up to Python 3.11.9)
- FFmpeg (for video processing)
- Windows OS (current configuration)

### Required Python Packages
```bash
pip install opencv-python mediapipe yt-dlp rich numpy matplotlib plotly scipy
```

### FFmpeg Setup
1. Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg\ffmpeg-8.0-essentials_build\`
3. Update the paths in the script if your installation differs:
   ```python
   ffmpeg_path = r"C:\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"
   ffplay_path = r"C:\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffplay.exe"
   ```

### Directory Structure
The tool automatically creates the following directory structure:
```
C:\Users\[username]\Hand Tracking Mediapipe\
├── videos/           # Downloaded/input videos
├── tracked/          # Temporarily processed videos
├── csv_data/         # Hand tracking data exports
├── reports/          # Analysis reports and visualizations
└── url_cache.txt     # Cached video URLs
```

## 🚀 Usage

### Quick Start
1. Run the script:
   ```bash
   python hand_tracking_tool.py
   ```

2. Choose input method:
   - **YouTube URL**: Enter any YouTube video URL
   - **Local File**: Enter filename or select by number
   - **Cached Video**: Reprocess previously downloaded videos

3. The tool will:
   - Download/process the video
   - Perform hand tracking analysis
   - Generate visualizations and reports
   - Display results in your preferred format

### Input Options
- **YouTube URLs**: Direct video processing from YouTube
- **Video Selection**: Choose from previously downloaded videos
- **File Management**: Delete videos or clear cache as needed

### Output Formats
- **CSV Data**: Frame-by-frame hand tracking coordinates
- **3D Trajectory**: Interactive HTML visualization
- **Analysis Reports**: Comprehensive text reports
- **Video Playback**: Side-by-side or tracked-only viewing

## 📊 Analysis Features

### Movement Statistics
- **Distance Tracking**: Total distance traveled by each hand
- **Speed Analysis**: Average, maximum, and minimum speeds
- **Position Analytics**: Movement ranges and center of mass
- **Detection Rates**: Frame-by-frame tracking success rates

### Visualization Types
1. **ASCII Heatmaps**: Quick CLI visualization of movement patterns
2. **3D Trajectories**: Interactive browser-based 3D movement paths
3. **Statistical Reports**: Detailed numerical analysis

### Data Export
- **CSV Format**: Frame, hand type, coordinates (x, y, z), landmark count
- **HTML Reports**: Interactive 3D visualizations
- **Text Reports**: Comprehensive statistical summaries

## 🎯 Possible Advanced Use Cases with Tampering

### Sports Analysis
- Golf swing analysis
- Tennis technique evaluation
- Baseball pitching mechanics
- Basketball shooting form

### Medical & Therapy
- Hand coordination assessment
- Rehabilitation progress tracking
- Motor skill evaluation
- Tremor analysis

### Research & Education
- Gesture recognition research
- Human-computer interaction studies
- Sign language analysis
- Movement pattern research

### Entertainment & Art
- Dance choreography analysis
- Musical performance evaluation
- Animation reference tracking
- Creative movement studies

## 📈 Technical Details

### Hand Detection Specifications
- **Framework**: Google MediaPipe
- **Max Hands**: 4 simultaneous detections
- **Confidence Threshold**: 0.5 (adjustable)
- **Landmark Points**: 21 per hand
- **Coordinate System**: Normalized 3D coordinates

### Performance Optimizations
- **Smart Caching**: Avoid re-downloading videos
- **Efficient Processing**: Frame-by-frame analysis with progress tracking
- **Memory Management**: Automatic cleanup of temporary files
- **Batch Processing**: Handle multiple videos efficiently

### Output Data Structure
```csv
frame,hand,wrist_x,wrist_y,wrist_z,num_landmarks
1,Left,0.456,0.234,0.123,21
1,Right,0.678,0.345,0.234,21
```

## 🔧 Configuration

### Customizable Parameters
```python
# Hand tracking sensitivity
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

# Maximum hands to detect
max_num_hands = 4

# Video processing quality
scale_factor = "iw/2:ih/2"  # Resize for faster processing
```

### Path Configuration
Update these paths based on your system setup:
```python
project_folder = r"C:\Users\ozzal\Hand Tracking Mediapipe"
ffmpeg_path = r"C:\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"
```

## 🚨 Troubleshooting

### Common Issues

**FFmpeg Not Found**
- Ensure FFmpeg is installed and paths are correct
- Verify FFmpeg is in your system PATH

**Video Download Fails**
- Check internet connection
- Verify YouTube URL is accessible
- Try clearing URL cache with `clear cache` command

**Low Detection Rates**
- Ensure hands are clearly visible in video
- Adjust `min_detection_confidence` parameter
- Check video quality and lighting

**Memory Issues**
- Process shorter videos for testing
- Increase system RAM if processing large files
- Check available disk space

### Performance Tips
- Use lower resolution videos for faster processing
- Ensure good lighting in source videos
- Keep hands clearly visible and unobstructed
- Process shorter clips for initial testing

## 📋 Requirements

### System Requirements
- **OS**: Windows (current configuration)
- **RAM**: 4GB+ recommended
- **Storage**: 1GB+ free space for video processing
- **CPU**: Multi-core processor recommended

### Python Dependencies
```txt
opencv-python>=4.5.0
mediapipe>=0.8.0
yt-dlp>=2023.1.6
rich>=12.0.0
numpy>=1.21.0
matplotlib>=3.5.0
plotly>=5.0.0
scipy>=1.7.0
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Cross-platform compatibility (macOS/Linux)
- Real-time webcam processing
- Additional analysis metrics
- Export format options
- GUI interface development

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Google MediaPipe**: For the excellent hand tracking framework
- **Rich Library**: For the beautiful CLI interface
- **Plotly**: For interactive 3D visualizations
- **yt-dlp**: For reliable video downloading

## 📞 Support

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Review existing issues in the repository
3. Create a detailed issue report with:
   - System information
   - Error messages
   - Steps to reproduce
   - Sample video (if applicable)

---

**Made with ❤️ for hand tracking analysis and movement research**
