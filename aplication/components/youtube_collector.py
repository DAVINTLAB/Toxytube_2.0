"""
YouTube Comments Collection Backend
Contains all business logic for collecting and processing YouTube comments
"""
import pandas as pd
import os
import re
from datetime import datetime


# =============================================================================
# Video ID and URL Processing
# =============================================================================

def extractVideoId(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        r'youtube\.com/watch\?.*v=([^&\n?#]+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# =============================================================================
# Video Information Retrieval
# =============================================================================

def getVideoInfo(videoId, apiKey):
    """Get video information from YouTube API"""
    if not apiKey.strip():
        return None, "API key not provided"

    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError

        # Build YouTube API service
        youtube = build('youtube', 'v3', developerKey=apiKey)

        # Get video info
        response = youtube.videos().list(
            part='snippet,statistics,contentDetails',
            id=videoId
        ).execute()

        if not response['items']:
            return None, "Video not found"

        video = response['items'][0]
        return video, None

    except ImportError:
        return None, "google-api-python-client not installed"
    except HttpError as e:
        return None, f"API error: {e.reason}"
    except Exception as e:
        return None, f"Error: {str(e)}"


# =============================================================================
# Formatting Utilities
# =============================================================================

def formatNumber(num):
    """Format large numbers with K, M, B suffixes"""
    if num is None:
        return "N/A"

    num = int(num)
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:,}"


def formatDuration(duration):
    """Format ISO 8601 duration to readable format"""
    if not duration:
        return "N/A"

    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
    if not match:
        return "N/A"

    hours, minutes, seconds = match.groups()
    hours = int(hours or 0)
    minutes = int(minutes or 0)
    seconds = int(seconds or 0)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


# =============================================================================
# Comment Collection
# =============================================================================

def collectComments(videoId, apiKey, maxComments=100, progressCallback=None):
    """
    Collect comments from a YouTube video
    
    Args:
        videoId: YouTube video ID
        apiKey: YouTube Data API v3 key
        maxComments: Maximum number of comments to collect
        progressCallback: Function to call with progress updates (message, collected_count)
        
    Returns:
        dict with:
            - success: bool
            - comments: list of comment dicts (if successful)
            - videoInfo: video information dict (if successful)
            - error: error message (if failed)
            - errorType: type of error ('quota', 'notfound', 'disabled', 'api', 'unknown')
    """
    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
    except ImportError:
        return {
            'success': False,
            'error': 'Biblioteca necessária faltando: google-api-python-client',
            'errorType': 'import'
        }

    try:
        # Build YouTube API service
        if progressCallback:
            progressCallback("🔌 Connecting to YouTube API...", 0)

        youtube = build('youtube', 'v3', developerKey=apiKey)

        if progressCallback:
            progressCallback("✅ Successfully connected to YouTube API", 0)

        # Get video info
        if progressCallback:
            progressCallback("📹 Fetching video information...", 0)

        videoResponse = youtube.videos().list(
            part='snippet,statistics',
            id=videoId
        ).execute()

        if not videoResponse['items']:
            return {
                'success': False,
                'error': 'Vídeo não encontrado ou está privado',
                'errorType': 'notfound'
            }

        videoInfo = videoResponse['items'][0]
        videoTitle = videoInfo['snippet']['title']
        videoStats = videoInfo['statistics']
        commentCount = int(videoStats.get('commentCount', 0))

        if progressCallback:
            progressCallback(f"✅ Video found: {videoTitle[:60]}...", 0)
            progressCallback(f"💬 Total comments available: {commentCount:,}", 0)

        # Collect comments
        if progressCallback:
            progressCallback("📝 Starting comment collection...", 0)

        comments = []
        nextPageToken = None
        collectedCount = 0

        while collectedCount < maxComments:
            # Calculate remaining comments to fetch
            remainingComments = min(100, maxComments - collectedCount)

            # Fetch comments
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=videoId,
                maxResults=remainingComments,
                pageToken=nextPageToken,
                order='time'
            ).execute()

            # Process comments
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']

                comments.append({
                    'comment_id': item['snippet']['topLevelComment']['id'],
                    'author': comment['authorDisplayName'],
                    'comment_text': comment['textDisplay'],
                    'like_count': comment['likeCount'],
                    'published_at': comment['publishedAt'],
                    'updated_at': comment['updatedAt'],
                    'reply_count': item['snippet']['totalReplyCount']
                })

                collectedCount += 1
                if collectedCount >= maxComments:
                    break

            # Update progress (handled elsewhere)
            if progressCallback:
                pass

            # Check for next page
            nextPageToken = response.get('nextPageToken')
            if not nextPageToken:
                if progressCallback:
                    progressCallback("✅ End of available comments reached", collectedCount)
                break

        if not comments:
            return {
                'success': False,
                'error': 'No comments found or comments are disabled',
                'errorType': 'disabled'
            }

        return {
            'success': True,
            'comments': comments,
            'videoInfo': videoInfo,
            'videoStats': videoStats,
            'totalCollected': len(comments)
        }

    except HttpError as e:
        errorType = 'api'
        errorMsg = str(e)

        if e.resp.status == 403:
            errorType = 'quota'
            errorMsg = 'Daily quota limit reached. Try again tomorrow or use a different API key'

        return {
            'success': False,
            'error': errorMsg,
            'errorType': errorType
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'errorType': 'unknown'
        }


# =============================================================================
# File Saving
# =============================================================================

def saveCommentsToFile(comments, outputFilePath, outputFormat='csv'):
    """
    Save comments to file in specified format
    
    Args:
        comments: List of comment dictionaries
        outputFilePath: Full path where to save the file
        outputFormat: Format to save ('csv', 'xlsx', 'json', 'parquet')
        
    Returns:
        dict with:
            - success: bool
            - filePath: path to saved file (if successful)
            - fileSize: size of file in bytes (if successful)
            - error: error message (if failed)
            - fallbackFormat: format used if original format failed
    """
    try:
        # Create DataFrame
        df = pd.DataFrame(comments)

        # Convert date format
        df['published_at'] = pd.to_datetime(df['published_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['updated_at'] = pd.to_datetime(df['updated_at']).dt.strftime('%Y-%m-%d %H:%M:%S')

        # Save to selected format
        actualFilePath = outputFilePath
        actualFormat = outputFormat

        try:
            if outputFormat == 'csv':
                df.to_csv(outputFilePath, index=False, encoding='utf-8')
            elif outputFormat == 'xlsx':
                df.to_excel(outputFilePath, index=False, engine='openpyxl')
            elif outputFormat == 'json':
                df.to_json(outputFilePath, orient='records', indent=2, force_ascii=False)
            elif outputFormat == 'parquet':
                df.to_parquet(outputFilePath, index=False)
            else:
                # Fallback to CSV
                actualFilePath = outputFilePath.replace(f'.{outputFormat}', '.csv')
                df.to_csv(actualFilePath, index=False, encoding='utf-8')
                actualFormat = 'csv'

        except ImportError:
            # Handle missing dependencies - fallback to CSV
            actualFilePath = outputFilePath.replace(f'.{outputFormat}', '.csv')
            df.to_csv(actualFilePath, index=False, encoding='utf-8')
            actualFormat = 'csv'

        # Verify file was created
        if os.path.exists(actualFilePath):
            fileSize = os.path.getsize(actualFilePath)

            result = {
                'success': True,
                'filePath': actualFilePath,
                'fileSize': fileSize
            }

            if actualFormat != outputFormat:
                result['fallbackFormat'] = actualFormat

            return result
        else:
            return {
                'success': False,
                'error': 'O arquivo não foi criado com sucesso'
            }

    except Exception as e:
        return {
            'success': False,
            'error': f'Erro ao salvar arquivo: {str(e)}'
        }


# =============================================================================
# Directory Management
# =============================================================================

def prepareOutputDirectory(outputDirectory):
    """
    Prepare output directory, create if needed, test permissions
    
    Args:
        outputDirectory: Path to directory
        
    Returns:
        dict with:
            - success: bool
            - directory: validated directory path (if successful)
            - created: whether directory was created
            - usedFallback: whether fallback directory was used
            - error: error message (if failed)
    """
    try:
        import pathlib

        # Normalize path
        outputDirectory = os.path.normpath(outputDirectory.strip())

        # Create directory if it doesn't exist
        created = False
        if not os.path.exists(outputDirectory):
            os.makedirs(outputDirectory, exist_ok=True)
            created = True

        # Test write permissions
        testFile = os.path.join(outputDirectory, "test_write.tmp")
        usedFallback = False

        try:
            with open(testFile, 'w') as f:
                f.write("test")
            os.remove(testFile)
        except PermissionError:
            # Fallback to Documents
            fallbackDir = pathlib.Path.home() / "Documents" / "YouTube_Comments"
            fallbackDir.mkdir(parents=True, exist_ok=True)
            outputDirectory = str(fallbackDir)
            usedFallback = True

        return {
            'success': True,
            'directory': outputDirectory,
            'created': created,
            'usedFallback': usedFallback
        }

    except Exception as e:
        return {
            'success': False,
            'error': f'Erro ao preparar diretório: {str(e)}'
        }
