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
        'outputFormat': 'csv',              # File format (csv, xlsx, json, parquet)
        'maxComments': 100,                 # Maximum number of comments to collect
        'isExecuting': False,               # Execution status
        'collectionResults': None,          # Collection results
        'videoCommentCount': 0,             # Total comments available in video
        'videoTitle': ''                    # Video title
    }

# =============================================================================
# Helper Functions - Quota Estimation
# =============================================================================

def sanitize_filename(filename):
    """
    Remove or replace characters that are not allowed in filenames.
    
    Args:
        filename: Original filename string
        
    Returns:
        Sanitized filename safe for all operating systems
    """
    # Replace invalid characters with underscore
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    # Replace multiple spaces with single underscore
    import re
    filename = re.sub(r'\s+', '_', filename)

    # Remove leading/trailing underscores and dots
    filename = filename.strip('_.')

    # Limit length to 200 characters
    if len(filename) > 200:
        filename = filename[:200]

    return filename

def estimate_quota_cost(num_comments):
    """
    Estimate API quota cost for collecting comments.
    
    YouTube API Quota costs:
    - videos.list: 1 unit
    - commentThreads.list: 1 unit per request (max 100 comments per request)
    
    Args:
        num_comments: Number of comments to collect
        
    Returns:
        dict with quota estimation details
    """
    # Initial video info request
    video_info_cost = 1

    # Comment requests (100 comments per request, 1 unit each)
    num_requests = (num_comments + 99) // 100  # Ceiling division
    comments_cost = num_requests

    total_cost = video_info_cost + comments_cost

    return {
        'video_info_cost': video_info_cost,
        'comments_cost': comments_cost,
        'num_requests': num_requests,
        'total_cost': total_cost,
        'percentage_of_daily': (total_cost / 10000) * 100
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

                        # Store video title in session state
                        st.session_state.youtubeTask['videoTitle'] = title

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
                            # Store comment count in session state
                            st.session_state.youtubeTask['videoCommentCount'] = int(comments)
                        else:
                            st.warning("⚠️ Comments may be disabled")
                            st.session_state.youtubeTask['videoCommentCount'] = 0

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
                        st.info("💡 Configure a valid API key to see more information")

                else:
                    # Show basic info without API
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

st.markdown("")

# =============================================================================
# Step 2: Collection Settings
# =============================================================================

# Recalculate step1_complete after session state updates
step1_complete = (
    st.session_state.youtubeTask['apiKey'].strip() != '' and
    st.session_state.youtubeTask['videoId'] is not None and
    st.session_state.youtubeTask['videoId'] != ''
)

# Check if step 2 is complete
step2_complete = st.session_state.youtubeTask['maxComments'] > 0

with st.container(border=True):
    st.markdown("### 📊 Step 2: Collection Settings")
    st.markdown("Configure how many comments to collect and estimate API quota usage.")

    if step1_complete:
        video_comment_count = st.session_state.youtubeTask.get('videoCommentCount', 0)

        # Collect All checkbox
        if video_comment_count > 0:
            collect_all = st.checkbox(
                "📥 Collect All Comments",
                value=True,
                help=f"Automatically collect all {video_comment_count:,} comments from the video"
            )
        else:
            collect_all = False

        # Create columns: input field and quota estimation
        input_col, quota_col1, quota_col2, quota_col3 = st.columns([2, 1.5, 1.5, 1.5])

        with input_col:
            if video_comment_count > 0:
                max_available = min(video_comment_count, 10000)

                # If collect_all is checked, set to max_available and disable input
                if collect_all:
                    maxComments = max_available
                    st.number_input(
                        "Number of Comments to Collect:",
                        min_value=1,
                        max_value=max_available,
                        value=max_available,
                        disabled=True,
                        help=f"Collecting all available comments: {max_available:,}"
                    )
                else:
                    maxComments = st.number_input(
                        "Number of Comments to Collect:",
                        min_value=1,
                        max_value=max_available,
                        value=min(st.session_state.youtubeTask['maxComments'], max_available),
                        help=f"Enter number of comments to collect (max available: {video_comment_count:,})"
                    )

                # Show percentage of total
                percentage_selected = (maxComments / video_comment_count) * 100
                st.caption(f"{maxComments:,} of {video_comment_count:,} comments ({percentage_selected:.1f}%)")
            else:
                maxComments = st.number_input(
                    "Maximum Comments to Collect:",
                    min_value=1,
                    max_value=10000,
                    value=st.session_state.youtubeTask['maxComments'],
                    help="Maximum number of comments to collect"
                )

        # Update max comments
        st.session_state.youtubeTask['maxComments'] = maxComments

        # API Quota estimation in columns next to input
        quota_info = estimate_quota_cost(maxComments)

        with quota_col1:
            st.metric(
                "API Requests",
                f"{quota_info['num_requests'] + 1}",
                help="Number of API requests needed"
            )

        with quota_col2:
            st.metric(
                "Quota Cost",
                f"{quota_info['total_cost']:,}",
                help="Total quota units consumed"
            )

        with quota_col3:
            st.metric(
                "Daily Usage",
                f"{quota_info['percentage_of_daily']:.1f}%",
                help="% of daily quota (10,000 units)"
            )

        st.markdown("")

        # Quota information
        st.info("💡 **YouTube API Quota:** Daily limit of 10,000 units (resets at midnight Pacific Time). Video info costs 1 unit, comments cost 1 unit per 100 comments.")

        # Important note about comment collection
        st.warning("⚠️ **Note:** Only top-level comments are collected (replies are excluded). The final count may be lower than the video's total comment count.")
    else:
        st.info("💡 Complete Step 1 to configure collection settings.")

st.markdown("")

# =============================================================================
# Step 3: Output Configuration
# =============================================================================

# Check if step 3 is complete
step3_complete = st.session_state.youtubeTask['outputFileName'].strip() != ''

with st.container(border=True):
    st.markdown("### 💾 Step 3: Output Configuration")
    st.markdown("Configure the file format for saving the comments.")

    # Set output directory to Downloads by default
    outputDirectory = str(pathlib.Path.home() / "Downloads")
    st.session_state.youtubeTask['outputDirectory'] = outputDirectory

    # Generate default file name from video title
    video_title = st.session_state.youtubeTask.get('videoTitle', '')
    if video_title:
        sanitized_title = sanitize_filename(video_title)
        defaultFileName = f"{sanitized_title}__comments"
    else:
        defaultFileName = f"comments_{st.session_state.youtubeTask.get('videoId', 'unknown')}_{datetime.now().strftime('%d-%m-%Y')}"

    # Auto-update filename when video title is available
    current_filename = st.session_state.youtubeTask.get('outputFileName', '')
    if video_title and (not current_filename or current_filename.startswith('comments_')):
        # Update to use video title if we have it and current name is generic
        st.session_state.youtubeTask['outputFileName'] = defaultFileName

    # Create two columns for file name and format
    nameCol, formatCol = st.columns([8.5, 1.5])

    with nameCol:
        outputFileName = st.text_input(
            "File name (without extension):",
            value=st.session_state.youtubeTask.get('outputFileName', defaultFileName),
            placeholder=defaultFileName,
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

    # Show preview of full file name and path
    if outputFileName.strip():
        fullFileName = f"{outputFileName.strip()}.{outputFormat}"
        full_path = os.path.join(outputDirectory, fullFileName)
        st.info(f"💡 **File will be saved to**: {full_path}")

    # Check if configuration is complete
    if not outputFileName.strip():
        st.warning("⚠️ Please enter a file name to continue.")

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
            st.success(f"✅ Directory created: {outputDirectory}")

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
            st.info(f"💡 **Comments Location**: {results['outputPath']}")

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