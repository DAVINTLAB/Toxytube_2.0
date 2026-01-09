"""
YouTube Comments Collection Page
Collects comments from YouTube videos via API
"""
import streamlit as st
import pandas as pd
import os
from datetime import datetime
import pathlib
import re
import sys

# Add parent directory to path to import components

from components.navigation import render_navigation
from components.youtube_collector import (
    extractVideoId,
    getVideoInfo,
    formatNumber,
    formatDuration,
    collectComments,
    saveCommentsToFile,
    prepareOutputDirectory
)

# =============================================================================
# Initialize Session State - Collection Process Data
# =============================================================================

# Dictionary with all data needed for the YouTube comment collection process
if 'youtubeTask' not in st.session_state:
    st.session_state.youtubeTask = {
        'videoUrl': '',                     # YouTube video URL
        'videoId': '',                      # Extracted video ID
        'apiKey': '',                       # YouTube API key
        'outputDirectory': '',              # Output directory
        'outputFileName': '',               # Output file name
        'outputFormat': 'csv',              # File format (csv, xlsx, json)
        'maxComments': 100,                 # Maximum number of comments to collect
        'isExecuting': False,               # Execution status
        'collectionResults': None           # Collection results
    }

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="YouTube Comments",
    page_icon="🎥",
    layout="wide",
)

# Render navigation sidebar
render_navigation('youtube')

# =============================================================================
# Main Content
# =============================================================================

# Page header
st.markdown("# 🎥 YouTube Comment Collection")
st.markdown("Collect comments from YouTube videos using the YouTube API in a simple and intuitive way.")

st.markdown("---")



# =============================================================================
# Step 1: Video Selection and API Configuration
# =============================================================================

# Check if step 1 is complete
step1_complete = (
    st.session_state.youtubeTask['apiKey'].strip() != '' and
    st.session_state.youtubeTask['videoId'] is not None and
    st.session_state.youtubeTask['videoId'] != ''
)

with st.container(border=True):
    st.markdown("### 🎬 Step 1: API Configuration and Video Selection")
    st.markdown("Configure your YouTube API key and select the video.")

    # YouTube API Key input
    apiKey = st.text_input(
        "YouTube API Key:",
        value=st.session_state.youtubeTask['apiKey'],
        type="password",
        placeholder="Enter your YouTube Data API v3 key",
        help="You need a YouTube Data API v3 key from Google Cloud Console"
    )

    # Update API key
    st.session_state.youtubeTask['apiKey'] = apiKey

    if not apiKey:
        st.info("💡 **Need an API key?** Go to [Google Cloud Console](https://console.cloud.google.com/) → APIs & Services → Credentials → Create API Key → Enable YouTube Data API v3")

    # YouTube video URL input
    videoUrl = st.text_input(
        "YouTube Video URL:",
        value=st.session_state.youtubeTask['videoUrl'],
        placeholder="https://www.youtube.com/watch?v=VIDEO_ID",
        help="Paste the full YouTube video URL here"
    )

    # Update video URL in task
    st.session_state.youtubeTask['videoUrl'] = videoUrl

    # Extract and validate video ID
    if videoUrl.strip():
        videoId = extractVideoId(videoUrl)
        st.session_state.youtubeTask['videoId'] = videoId

        if videoId:
            # Video preview and information

            # Create two columns: one for video, one for info
            videoCol, infoCol = st.columns([3, 2])

            with videoCol:
                st.markdown("#### Video Preview")
                try:
                    st.video(videoUrl)
                except Exception as e:
                    st.warning("⚠️ Could not load video preview, but the ID is valid.")

            with infoCol:
                st.markdown("#### Video Information")

                if apiKey.strip():
                    # Get video info from API
                    videoInfo, error = getVideoInfo(videoId, apiKey)

                    if videoInfo and not error:
                        snippet = videoInfo.get('snippet', {})
                        statistics = videoInfo.get('statistics', {})
                        contentDetails = videoInfo.get('contentDetails', {})

                        # Video title
                        title = snippet.get('title', 'Title not available')
                        st.info(f"**📝 Title**: {title[:100]}")

                        # Channel name
                        channel = snippet.get('channelTitle', 'Channel not available')
                        st.info(f"**👤 Channel**: {channel}")

                        # View count
                        views = statistics.get('viewCount')
                        if views:
                            st.metric("👁️ Views", formatNumber(views))

                        # Like count
                        likes = statistics.get('likeCount')
                        if likes:
                            st.metric("👍 Likes", formatNumber(likes))

                        # Comment count
                        comments = statistics.get('commentCount')
                        if comments:
                            st.metric("💬 Comments", formatNumber(comments))
                        else:
                            st.warning("⚠️ Comments may be disabled")

                        # Video duration
                        duration = contentDetails.get('duration')
                        if duration:
                            st.info(f"**⏱️ Duration**: {formatDuration(duration)}")

                        # Publication date
                        publishedAt = snippet.get('publishedAt')
                        if publishedAt:
                            from datetime import datetime
                            pubDate = datetime.fromisoformat(publishedAt.replace('Z', '+00:00'))
                            st.info(f"**📅 Published**: {pubDate.strftime('%d/%m/%Y')}")

                        # # Video description preview
                        # description = snippet.get('description', '')
                        # if description:
                        #     st.info(f"**📄 Descrição**: {description[:100]}...")

                    else:
                        st.warning(f"⚠️ Could not get detailed information: {error}")
                        # Show basic info
                        st.info(f"**🆔 Video ID**: {videoId}")
                        st.info("💡 Configure a valid API key to see more information")

                else:
                    # Show basic info without API
                    st.info(f"**🆔 Video ID**: {videoId}")
                    st.info("💡 Configure the API to see detailed information such as:")
                    st.markdown("""
                - 👁️ Views
                - 👍 Likes  
                - 💬 Number of comments
                - ⏱️ Video duration
                - 📅 Publication date
                - 👤 Channel name
                """)

        else:
            st.error("❌ Invalid YouTube URL. Check the URL format.")
    else:
        st.warning("⚠️ Enter the YouTube video URL to continue.")

    # Maximum comments to fetch
    maxComments = st.number_input(
        "Maximum Comments to Collect:",
        min_value=1,
        max_value=10000,
        value=st.session_state.youtubeTask['maxComments'],
        help="Maximum number of comments to collect (the API has rate limits)"
    )

    # Update max comments
    st.session_state.youtubeTask['maxComments'] = maxComments

    # API Quota information
    st.info("💡 **Important**: The YouTube API has a quota system that limits the number of comments that can be collected per day. Each request consumes part of your daily quota. More information about quota costs [here](https://developers.google.com/youtube/v3/determine_quota_cost).")

    # Step 1 completion indicator
    if step1_complete:
        st.success("✅ **Step 1 completed!** API configured and video selected.")

st.markdown("")

# =============================================================================
# Step 3: Output Configuration
# =============================================================================

# Check if step 3 is complete
step3_complete = (
    st.session_state.youtubeTask['outputDirectory'].strip() != '' and
    st.session_state.youtubeTask['outputFileName'].strip() != ''
)

with st.container(border=True):
    st.markdown("### 💾 Step 3: Output Configuration")
    st.markdown("Configure where and how the comments will be saved.")

    # Default path suggestions
    defaultPaths = [
        str(pathlib.Path.home() / "Desktop"),
        str(pathlib.Path.home() / "Documents"),
        str(pathlib.Path.home() / "Downloads"),
    ]

    # Get current suggested path
    currentSuggestion = st.session_state.youtubeTask['outputDirectory'] or defaultPaths[0]

    # Create columns for text input and buttons
    textCol, btn1Col, btn2Col, btn3Col = st.columns([8.5, 0.5, 0.5, 0.5])

    with textCol:
        outputDirectory = st.text_input(
            label="Output Directory:",
            value=currentSuggestion,
            placeholder="Ex: C:/Users/username/Desktop/youtube_comments/",
            help="Directory where the file will be saved",
        )

    with btn1Col:
        # Adiciona um pequeno espaço no topo para alinhar
        st.markdown("<div style='margin-top: 27px;'></div>", unsafe_allow_html=True)
        if st.button("📋", help="Desktop", use_container_width=True):
            st.session_state.youtubeTask['outputDirectory'] = defaultPaths[0]
            st.rerun()

    with btn2Col:
        st.markdown("<div style='margin-top: 27px;'></div>", unsafe_allow_html=True)
        if st.button("📁", help="Documents", use_container_width=True):
            st.session_state.youtubeTask['outputDirectory'] = defaultPaths[1]
            st.rerun()

    with btn3Col:
        st.markdown("<div style='margin-top: 27px;'></div>", unsafe_allow_html=True)
        if st.button("⬇️", help="Downloads", use_container_width=True):
            st.session_state.youtubeTask['outputDirectory'] = defaultPaths[2]
            st.rerun()

    # Update output directory
    st.session_state.youtubeTask['outputDirectory'] = outputDirectory

    # Create two columns for file name and format
    nameCol, formatCol = st.columns([8.5, 1.5])

    with nameCol:
        # Default file name with timestamp
        defaultFileName = f"comments_{st.session_state.youtubeTask.get('videoId', 'unknown')}_{datetime.now().strftime('%d-%m-%Y')}"

        outputFileName = st.text_input(
            "File name (without extension):",
            value=st.session_state.youtubeTask.get('outputFileName') or defaultFileName,
            placeholder=f"comments_{st.session_state.youtubeTask.get('videoId', 'unknown')}_{datetime.now().strftime('%d-%m-%Y')}",
            help="Enter the desired file name"
        )

        # Update output file name
        st.session_state.youtubeTask['outputFileName'] = outputFileName

    with formatCol:
        # File format selector
        outputFormat = st.selectbox(
            "File format:",
            options=['csv', 'xlsx', 'json', 'parquet'],
            index=['csv', 'xlsx', 'json', 'parquet'].index(st.session_state.youtubeTask.get('outputFormat', 'csv')),
            help="Choose the data export format"
        )

        # Update output format
        st.session_state.youtubeTask['outputFormat'] = outputFormat

    # Show preview of full file name
    if outputFileName.strip():
        fullFileName = f"{outputFileName.strip()}.{outputFormat}"
        st.info(f"**💡 Final file with comments**: {fullFileName}")

    # Check if configuration is complete
    if not (outputDirectory.strip() and outputFileName.strip()):
        st.warning("⚠️ Fill in all fields to continue.")

    # Step 3 completion indicator
    if step3_complete:
        st.success("✅ **Step 3 completed!** Output configuration defined.")

st.markdown("")

# =============================================================================
# Step 4: Execution
# =============================================================================

# Check if step 4 is complete
step4_complete = (
    st.session_state.youtubeTask.get('collectionResults') is not None and
    st.session_state.youtubeTask['collectionResults'].get('success', False)
)

with st.container(border=True):
    st.markdown("### 🚀 Step 4: Collection Execution")
    st.markdown("Execute the video comment collection.")

    # Get task data
    videoUrl = st.session_state.youtubeTask['videoUrl']
    videoId = st.session_state.youtubeTask['videoId']
    outputDirectory = st.session_state.youtubeTask['outputDirectory']
    outputFileName = st.session_state.youtubeTask['outputFileName']
    outputFormat = st.session_state.youtubeTask['outputFormat']

    # Check execution state
    isExecuting = st.session_state.youtubeTask['isExecuting']

    # Execution button - can only execute if all fields are filled
    canExecute = (not isExecuting and
                 apiKey.strip() and
                 videoId and
                 outputDirectory.strip() and
                 outputFileName.strip())

    executeButton = st.button(
        "📥 Start Comment Collection",
        disabled=not canExecute,
        use_container_width=True,
        help="Start the YouTube comment collection process" if canExecute else "Fill in all fields from previous steps to execute"
    )

    # Stop execution button (only visible during execution)
    if isExecuting:
        if st.button("⏹️ Stop Collection", use_container_width=True, type="secondary"):
            st.session_state.youtubeTask['isExecuting'] = False
            st.warning("⚠️ Collection stopped by user.")
            st.rerun()

    # Execute comment collection
    if executeButton and canExecute:
        st.session_state.youtubeTask['isExecuting'] = True

        # Validate and create output directory
        dirResult = prepareOutputDirectory(outputDirectory)

        if not dirResult['success']:
            st.error(f"❌ {dirResult['error']}")
            st.session_state.youtubeTask['isExecuting'] = False
            st.stop()

        outputDirectory = dirResult['directory']

        if dirResult['created']:
            st.success(f"📁 Directory created: {outputDirectory}")

        if dirResult['usedFallback']:
            st.warning(f"⚠️ Permission denied. Using alternative directory: {outputDirectory}")

        # Create output file path
        fileName = f"{outputFileName.strip()}.{outputFormat}"
        outputFilePath = os.path.join(outputDirectory, fileName)

        # Create terminal output
        st.markdown("#### 💻 Collection Terminal")
        terminalOutput = st.empty()

        # Initialize terminal messages
        terminalMessages = ["⏳ Initializing API connection..."]
        terminalOutput.code("\n".join(terminalMessages), language=None)

        # Progress callback to update terminal
        def progressCallback(message, count):
            terminalMessages.append(message)
            terminalOutput.code("\n".join(terminalMessages), language="bash")

        # Collect comments using backend function
        collectionResult = collectComments(
            videoId=videoId,
            apiKey=apiKey,
            maxComments=maxComments,
            progressCallback=progressCallback
        )

        # Handle collection result
        if collectionResult['success']:
            comments = collectionResult['comments']
            videoInfo = collectionResult['videoInfo']
            videoTitle = videoInfo['snippet']['title']
            videoStats = collectionResult['videoStats']

            # Save comments to file
            terminalMessages.append(f"💾 Salvando comentários em {outputFormat.upper()}...")
            terminalOutput.code("\n".join(terminalMessages), language="bash")

            saveResult = saveCommentsToFile(comments, outputFilePath, outputFormat)

            if saveResult['success']:
                actualPath = saveResult['filePath']
                fileSize = saveResult['fileSize']

                # Handle format fallback
                if 'fallbackFormat' in saveResult:
                    terminalMessages.append(f"⚠️ Salvando como {saveResult['fallbackFormat'].upper()} (formato original não suportado)...")
                    terminalOutput.code("\n".join(terminalMessages), language="bash")

                finalMessages = terminalMessages + [
                    "",
                    "=" * 50,
                    "✅ COLETA DE COMENTÁRIOS CONCLUÍDA!",
                    f"📁 Arquivo salvo: {actualPath}",
                    f"📊 Comentários coletados: {len(comments):,}",
                    f"💾 Tamanho do arquivo: {fileSize / 1024:.1f} KB",
                    f"🎥 Vídeo: {videoTitle[:50]}...",
                    "=" * 50
                ]

                st.session_state.youtubeTask['collectionResults'] = {
                    'success': True,
                    'outputPath': actualPath,
                    'totalComments': len(comments),
                    'videoTitle': videoTitle,
                    'fileSize': fileSize,
                    'videoStats': videoStats
                }
            else:
                finalMessages = terminalMessages + [
                    "",
                    "❌ ERRO AO SALVAR ARQUIVO:",
                    saveResult['error']
                ]
                st.session_state.youtubeTask['collectionResults'] = {
                    'success': False,
                    'error': saveResult['error']
                }
        else:
            # Handle collection errors
            errorType = collectionResult['errorType']
            error = collectionResult['error']

            if errorType == 'import':
                finalMessages = terminalMessages + [
                    "",
                    "❌ MISSING DEPENDENCIES:",
                    error,
                    "",
                    "💡 Install with: pip install google-api-python-client"
                ]
                st.error("❌ Missing Dependencies")
                st.code("pip install google-api-python-client", language="bash")
            elif errorType == 'quota':
                finalMessages = terminalMessages + [
                    "",
                    "❌ API QUOTA EXCEEDED:",
                    error
                ]
            elif errorType == 'notfound':
                finalMessages = terminalMessages + [
                    "",
                    "❌ VIDEO NOT FOUND:",
                    error,
                    "Please check the URL and try again"
                ]
            elif errorType == 'disabled':
                finalMessages = terminalMessages + [
                    "",
                    "⚠️ NO COMMENTS COLLECTED",
                    error
                ]
            else:
                finalMessages = terminalMessages + [
                    "",
                    "❌ COLLECTION ERROR:",
                    error,
                    "",
                    "💡 Common issues:",
                    "- Invalid API key",
                    "- Video with disabled comments",
                    "- API quota exceeded",
                    "- Network connection problems"
                ]

            st.session_state.youtubeTask['collectionResults'] = {
                'success': False,
                'error': error
            }

        # Update final terminal display
        terminalOutput.code("\n".join(finalMessages), language="bash")

        # Update execution state
        st.session_state.youtubeTask['isExecuting'] = False

    # Show collection results
    if st.session_state.youtubeTask['collectionResults']:
        results = st.session_state.youtubeTask['collectionResults']

        if results['success']:
            st.success("✅ Comment collection completed successfully!")
            st.markdown("---")
            # Show results summary
            st.markdown("#### 📊 Collection Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                if 'videoStats' in results:
                    totalComments = int(results['videoStats'].get('commentCount', 0))
                    st.metric("💬 Total Video Comments", f"{totalComments:,}")

            with col2:
                st.metric("📥 Comments Collected", f"{results['totalComments']:,}")

            with col3:
                st.metric("💾 File Size", f"{results['fileSize'] / 1024:.1f} KB")

            # Show video info
            st.info(f"**💡 Comments Location**: {results['outputPath']}")

            # Show sample data if available
            if os.path.exists(results['outputPath']):
                st.markdown("#### 📋 Comment Sample")
                try:
                    # Read sample data based on file extension
                    filePath = results['outputPath']
                    fileExtension = filePath.split('.')[-1].lower()

                    if fileExtension == 'csv':
                        sampleDf = pd.read_csv(filePath).head(5)
                    elif fileExtension == 'xlsx':
                        sampleDf = pd.read_excel(filePath).head(5)
                    elif fileExtension == 'json':
                        sampleDf = pd.read_json(filePath).head(5)
                    elif fileExtension == 'parquet':
                        sampleDf = pd.read_parquet(filePath).head(5)
                    else:
                        # Try CSV as fallback
                        sampleDf = pd.read_csv(filePath).head(5)

                    st.dataframe(sampleDf[['author', 'comment_text', 'published_at', 'like_count']], use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error loading sample: {str(e)}")
        else:
            st.error(f"❌ Collection failed: {results['error']}")

    # Step 4 completion indicator
    if step4_complete:
        st.success("✅ **Step 4 completed!** Comments collected and saved successfully.")