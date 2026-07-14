# Requirement Analysis Phase

## Functional Requirements
1. **Audio Ingestion**: Support `.wav` and `.mp3` format uploads up to 200MB.
2. **Transcription**: Output accurate, normalized textual transcripts of speech.
3. **Similarity Assessment**: Compare semantic overlap of transcripts against reference concepts.
4. **Fluency Analysis**: Extract ZCR, RMS energy, silence pauses, and filler counts.
5. **PDF Reporting**: Generate grid-lined summary tables and waveform graphs.
6. **Persistence**: Store results across a 10-table SQLite relational database schema.

## Non-Functional Requirements
- **Response Time**: Analysis of 30-second audio files should complete within 10 seconds.
- **Robustness**: Maintain local analysis fallback if the FastAPI backend service is offline.
- **Portability**: Compatible with Windows, Linux, and macOS.
- **UI Aesthetics**: Sleek, glassmorphism dark mode layouts with responsive layout structures.

## System Prerequisites
- **Language**: Python 3.10 or higher.
- **OS Tools**: `ffmpeg` binary added to the system `PATH`.
