import sqlite3
import random
import time
from datetime import datetime
import logging
import sys
import os
from typing import List, Dict, Optional, Union
import tweepy
from openai import OpenAI
import re
import httpx
import schedule
import threading
from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import InputPeerChannel, InputChannel
from telethon.tl.functions.channels import GetFullChannelRequest, GetChannelsRequest
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import secrets
import json
from pathlib import Path
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'twitter_poster_mcp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
    ]
)

logger = logging.getLogger("twitter-poster-mcp")

# API models
class TwitterCredentials(BaseModel):
    consumer_key: str
    consumer_secret: str
    access_token: str
    access_token_secret: str
    
class TelegramCredentials(BaseModel):
    api_id: int
    api_hash: str
    channel: str = "jinseBTC"  # Default channel

class AICredentials(BaseModel):
    api_key: str
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

class PostRequest(BaseModel):
    content: str
    hashtags: Optional[str] = "#crypto"

class ScheduleRequest(BaseModel):
    interval_minutes: int = 30
    min_interval_seconds: int = 2400  # Minimum interval between posts
    max_interval_seconds: int = 4800  # Maximum interval between posts
    
class ServiceConfig(BaseModel):
    service_name: str = "twitter-poster"
    twitter: TwitterCredentials
    telegram: TelegramCredentials
    ai: AICredentials
    prompt_template: str = """
    Please summarize the following messages from a cryptocurrency news channel into an engaging English tweet.
    If there are multiple messages, separate them with numbers 1, 2, 3, etc.
    Make the tweet appealing using cryptocurrency terminology and emojis.
    Keep the total length under 70 words.

    Original messages:
    {messages}
    """
    fixed_hashtags: str = "#crypto"

class UserConfig(BaseModel):
    api_key: str
    service_config: ServiceConfig

# Create FastAPI instance
app = FastAPI(
    title="Twitter Poster MCP",
    description="Microservice Control Panel for automatically posting Telegram messages to Twitter",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication
security = HTTPBearer()
API_KEYS = {}  # Store for API keys: {api_key: user_id}
USER_CONFIGS = {}  # Store for user configurations: {user_id: UserConfig}
USER_INSTANCES = {}  # Store for TwitterPoster instances: {user_id: TwitterPoster}
SCHEDULED_TASKS = {}  # Store for scheduled tasks: {user_id: task}

def load_configs():
    """Load existing configurations from disk"""
    config_path = Path("configs")
    config_path.mkdir(exist_ok=True)
    
    for config_file in config_path.glob("*.json"):
        try:
            with open(config_file, "r") as f:
                config_data = json.load(f)
                user_id = config_file.stem
                user_config = UserConfig(**config_data)
                USER_CONFIGS[user_id] = user_config
                API_KEYS[user_config.api_key] = user_id
                logger.info(f"Loaded configuration for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to load configuration {config_file}: {str(e)}")

def save_config(user_id: str, config: UserConfig):
    """Save user configuration to disk"""
    config_path = Path("configs")
    config_path.mkdir(exist_ok=True)
    
    with open(config_path / f"{user_id}.json", "w") as f:
        json.dump(config.dict(), f, indent=4)
    
    logger.info(f"Saved configuration for user {user_id}")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_key = credentials.credentials
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return API_KEYS[api_key]

class TwitterPoster:
    def __init__(self, config: ServiceConfig, user_id: str):
        self.user_id = user_id
        self.config = config
        self.logger = logging.getLogger(f"TwitterPoster-{user_id}")
        self.logger.info("Initializing Twitter Poster...")
        
        # Set up AI client
        try:
            # Get system proxy settings
            http_proxy = os.environ.get('http_proxy')
            https_proxy = os.environ.get('https_proxy')
            all_proxy = os.environ.get('all_proxy')
            
            self.logger.debug(f"Current system proxy settings: HTTP={http_proxy}, HTTPS={https_proxy}, ALL={all_proxy}")
            
            # Use proxy if available
            if https_proxy or http_proxy or all_proxy:
                proxy_url = https_proxy or http_proxy or all_proxy
                self.logger.debug(f"Using system proxy: {proxy_url}")
                transport = httpx.HTTPTransport(proxy=proxy_url)
                http_client = httpx.Client(transport=transport)
            else:
                # Direct connection without proxy
                self.logger.debug("No system proxy detected, using direct connection")
                http_client = httpx.Client()
            
            self.ai_client = OpenAI(
                api_key=config.ai.api_key,
                base_url=config.ai.base_url,
                http_client=http_client
            )
            self.logger.debug("OpenAI client setup successful")
        except Exception as e:
            self.logger.error(f"OpenAI client setup failed: {str(e)}")
            raise

        # Set up Twitter API client
        try:
            self.twitter_client = tweepy.Client(
                consumer_key=config.twitter.consumer_key,
                consumer_secret=config.twitter.consumer_secret,
                access_token=config.twitter.access_token,
                access_token_secret=config.twitter.access_token_secret
            )
            
            # Verify API permissions
            try:
                test_response = self.twitter_client.get_me()
                
                if hasattr(test_response, 'response') and hasattr(test_response.response, 'headers'):
                    headers = test_response.response.headers
                    rate_limit_remaining = headers.get('x-rate-limit-remaining')
                    rate_limit_reset = headers.get('x-rate-limit-reset')
                    rate_limit = headers.get('x-rate-limit-limit')
                    
                    if rate_limit_remaining and rate_limit and rate_limit_reset:
                        self.logger.info(f"API rate limit status: {rate_limit_remaining}/{rate_limit} remaining, reset at: {rate_limit_reset}")
                
                self.logger.debug("Twitter API permissions verified")
            except Exception as e:
                rate_limit_info = ""
                if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                    headers = e.response.headers
                    rate_limit_info = f"\nRequests remaining: {headers.get('x-rate-limit-remaining', 'unknown')}"
                    rate_limit_info += f"\nRate limit reset time: {headers.get('x-rate-limit-reset', 'unknown')}"
                
                raise Exception(f"Twitter API verification failed: {str(e)}{rate_limit_info}\nPlease ensure app permissions are correctly configured in the developer portal")

            self.logger.debug("Twitter API client setup successful")
        except Exception as e:
            self.logger.error(f"Twitter API client setup failed: {str(e)}")
            raise
        
        # Telegram configuration
        self.telegram_api_id = config.telegram.api_id
        self.telegram_api_hash = config.telegram.api_hash
        self.telegram_channel = config.telegram.channel
        self.last_message_date = None
        self.telegram_client = None
        self.channel_entity = None
        
        # Initialize Telegram client
        self._init_telegram_client()
        
        # Message processing prompt template
        self.prompt_template = config.prompt_template
        
        # Fixed hashtags
        self.fixed_hashtags = config.fixed_hashtags
        
        self.min_interval = 2400  # Minimum interval between posts (seconds)
        self.max_interval = 4800  # Maximum interval between posts (seconds)
        
        self.logger.info("Twitter Poster initialized successfully")
        self.is_running = False
        self.stop_event = threading.Event()

    def _init_telegram_client(self):
        """Initialize Telegram client"""
        try:
            self.logger.info("Initializing Telegram client...")
            
            # Configure proxy if needed
            proxy = None
            if os.environ.get('socks_proxy') or os.environ.get('all_proxy'):
                proxy_url = os.environ.get('socks_proxy') or os.environ.get('all_proxy')
                self.logger.info(f"Using proxy: {proxy_url}")
                proxy = {
                    'proxy_type': 'socks5',
                    'addr': '127.0.0.1',
                    'port': 7897,
                    'rdns': True
                }
            
            self.telegram_client = TelegramClient(
                f'twitter_poster_session_{self.user_id}', 
                self.telegram_api_id, 
                self.telegram_api_hash,
                proxy=proxy
            )
            
            # Start client
            self.telegram_client.start()
            self.logger.info("Telegram client initialized successfully")
            
            # Get channel entity
            self._get_channel_entity()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Telegram client: {str(e)}")
            raise

    def _get_channel_entity(self):
        """Get channel entity"""
        try:
            self.logger.debug(f"Attempting to get entity for channel @{self.telegram_channel}...")
            
            # Run async call correctly
            self.channel_entity = self.telegram_client.loop.run_until_complete(
                self.telegram_client.get_entity(f"@{self.telegram_channel}")
            )
            self.logger.info(f"Successfully retrieved channel entity: {self.channel_entity.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to get channel entity: {str(e)}")
            raise
    
    async def _fetch_messages_async(self, limit=10):
        """Async fetch channel messages"""
        try:
            if not self.channel_entity:
                await self._get_channel_entity()
                
            # Build request
            result = await self.telegram_client(GetHistoryRequest(
                peer=self.channel_entity,
                limit=limit,
                offset_date=None,
                offset_id=0,
                max_id=0,
                min_id=0,
                add_offset=0,
                hash=0
            ))
            
            return result.messages
        except Exception as e:
            self.logger.error(f"Async fetch messages failed: {str(e)}")
            return []

    def get_telegram_messages(self) -> str:
        """Get latest messages from Telegram channel, limited to most recent 5"""
        self.logger.debug(f"Getting messages from Telegram channel {self.telegram_channel}...")
        
        try:
            # Run async function to get messages
            messages = self.telegram_client.loop.run_until_complete(self._fetch_messages_async(20))
            
            if not messages:
                self.logger.info("No messages retrieved")
                return ""
                
            # Filter new messages
            new_messages = []
            latest_date = self.last_message_date
            
            for message in messages:
                # Check if message has text content
                if not hasattr(message, 'message') or not message.message:
                    continue
                    
                # Check if it's a new message
                if self.last_message_date is None or message.date > self.last_message_date:
                    new_messages.append(message.message)
                    
                    # Update latest message date
                    if latest_date is None or message.date > latest_date:
                        latest_date = message.date
            
            # Update latest message date
            if latest_date is not None and (self.last_message_date is None or latest_date > self.last_message_date):
                self.last_message_date = latest_date
                self.logger.debug(f"Updated latest message date: {self.last_message_date}")
            
            if not new_messages:
                self.logger.info("No new messages found")
                return ""
            
            # Only use the most recent 5 messages to control length
            recent_messages = new_messages[:5]
            self.logger.debug(f"Retrieved {len(new_messages)} new messages, using only the most recent {len(recent_messages)}")
            
            # Combine message text
            combined_messages = "\n\n".join(recent_messages)
            
            return combined_messages
            
        except Exception as e:
            self.logger.error(f"Error getting Telegram messages: {str(e)}")
            return ""

    def generate_post_content(self) -> str:
        """Generate tweet content"""
        self.logger.debug("Generating tweet content...")
        
        # Get Telegram messages
        messages = self.get_telegram_messages()
        if not messages:
            self.logger.warning("No Telegram messages retrieved, cannot generate content")
            return ""
        
        # Build prompt
        prompt = self.prompt_template.format(messages=messages)
        self.logger.debug(f"Using prompt to generate content: {prompt}")
        
        for attempt in range(3):
            try:
                self.logger.info("Preparing to send request to DeepSeek API...")
                completion = self.ai_client.chat.completions.create(
                    model="deepseek-r1",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    timeout=90.0  # 90-second timeout
                )
                
                self.logger.info("✅ Successfully sent prompt to DeepSeek and received response")
                
                content = completion.choices[0].message.content.strip()
                self.logger.info(f"✅ Successfully received content from DeepSeek, length: {len(content)} characters")
                self.logger.debug(f"API returned raw content: {content}")
                
                # Remove quotes and explanatory text
                content = re.sub(r'^["\']\s*|\s*["\']$', '', content)  # Remove leading/trailing quotes
                content = re.sub(r'\s*\（.*?\）\s*', '', content)      # Remove Chinese parentheses and contents
                content = re.sub(r'\s*\(.*?\)\s*', '', content)        # Remove English parentheses and contents
                content = re.sub(r'\*\(.*?\)\*', '', content)          # Remove parentheses with asterisks and contents
                content = content.strip()
                
                if content:
                    self.logger.debug(f"Cleaned tweet content: {content}")
                    return content
                else:
                    raise Exception("Cleaned content is empty")
                    
            except Exception as e:
                self.logger.error(f"Content generation attempt {attempt + 1} failed: {type(e).__name__}: {str(e)}")
                
                # Try to get more error information
                if hasattr(e, 'request'):
                    self.logger.error(f"Request URL: {e.request.url if hasattr(e.request, 'url') else 'unknown'}")
                    self.logger.error(f"Request method: {e.request.method if hasattr(e.request, 'method') else 'unknown'}")
                
                if hasattr(e, '__context__') and e.__context__ is not None:
                    self.logger.error(f"Underlying error: {type(e.__context__).__name__}: {str(e.__context__)}")
                
                if attempt < 2:
                    wait_time = 20 * (attempt + 1)  # 20s after first failure, 40s after second
                    self.logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
        
        # All attempts failed, return empty string
        self.logger.error("All generation attempts failed, returning empty")
        return ""

    def create_post(self, content: str, mention_users: List[str] = None, retries: int = 3) -> bool:
        """Send tweet using Twitter API"""
        if not content:
            self.logger.error("Content is empty, cannot send tweet")
            return False
            
        # Tweet content organization and truncation
        try:
            # If content includes multiple items, only select the first
            if content.startswith("1. ") and "\n\n2. " in content:
                first_item = content.split("\n\n2. ")[0].strip()
                self.logger.info(f"Content too long, only sending first item: {first_item}")
                content = first_item
            
            # Ensure content is within Twitter character limit (280 characters)
            if len(content) > 230:  # Leave space for hashtags
                content = content[:227] + "..."
                self.logger.info(f"Content truncated to below 280 characters: {content}")
        except Exception as e:
            self.logger.error(f"Error processing content: {str(e)}")
            # If processing fails, try simple truncation
            if len(content) > 230:
                content = content[:227] + "..."
        
        for attempt in range(retries):
            try:
                # Build complete tweet content, add fixed hashtags
                full_content = f"{content}\n\n{self.fixed_hashtags}"
                self.logger.debug(f"Complete tweet content: {full_content}")

                # Send tweet
                response = self.twitter_client.create_tweet(
                    text=full_content
                )
                
                if response and response.data:
                    self.logger.info(f"Tweet sent successfully, ID: {response.data['id']}")
                    return True
                else:
                    raise Exception("Tweet sending failed, no valid response received")
                    
            except Exception as e:
                self.logger.error(f"Failed to send tweet (attempt {attempt + 1}/{retries}): {str(e)}")
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 60  # Increasing wait time after each failure
                    self.logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    self.logger.error("Maximum retry attempts reached, sending failed")
                    return False
        return False

    def diagnose_network(self):
        """Diagnose network connection issues"""
        self.logger.info("Starting network diagnostics...")
        
        # Check for proxy environment variables
        proxies_env = {
            "http_proxy": os.environ.get("http_proxy", "None"),
            "https_proxy": os.environ.get("https_proxy", "None"),
            "all_proxy": os.environ.get("all_proxy", "None")
        }
        self.logger.info(f"System proxy environment variables: {proxies_env}")
        
        # Try DNS resolution
        try:
            import socket
            ip = socket.gethostbyname("dashscope.aliyuncs.com")
            self.logger.info(f"DNS resolution successful: dashscope.aliyuncs.com -> {ip}")
        except Exception as e:
            self.logger.error(f"DNS resolution failed: {str(e)}")
        
        # Try HTTP request test
        try:
            import requests
            # Request using system proxy
            response = requests.get("https://dashscope.aliyuncs.com", timeout=10)
            self.logger.info(f"HTTP request (using system proxy) successful: status code {response.status_code}")
        except Exception as e:
            self.logger.error(f"HTTP request (using system proxy) failed: {str(e)}")
        
        return True

    def scheduled_post(self):
        """Scheduled posting task"""
        if self.stop_event.is_set():
            self.logger.info("Stop event detected, ending scheduled task")
            return

        self.logger.info("Executing scheduled posting task...")
        try:
            # First run network diagnostics
            self.diagnose_network()
            
            content = self.generate_post_content()
            if not content:
                self.logger.error("Content generation failed, skipping this post")
                return
            
            if self.create_post(content):
                self.logger.info("Scheduled posting task completed successfully")
            else:
                self.logger.warning("Scheduled posting task failed")
                
        except Exception as e:
            self.logger.error(f"Error executing scheduled task: {str(e)}")

    def start_scheduler(self, interval_minutes=30):
        """Start scheduler with specified interval"""
        if self.is_running:
            self.logger.warning("Scheduler is already running")
            return False
            
        try:
            self.stop_event.clear()
            self.is_running = True
            
            # Run immediately first time
            threading.Thread(target=self.scheduled_post).start()
            
            # Set up recurring schedule
            schedule.every(interval_minutes).minutes.do(self.scheduled_post)
            
            # Start scheduler thread
            def run_scheduler():
                self.logger.info(f"Scheduler started with {interval_minutes} minute interval")
                while not self.stop_event.is_set():
                    schedule.run_pending()
                    time.sleep(1)
                self.logger.info("Scheduler stopped")
                
            threading.Thread(target=run_scheduler, daemon=True).start()
            return True
        except Exception as e:
            self.logger.error(f"Failed to start scheduler: {str(e)}")
            self.is_running = False
            return False

    def stop_scheduler(self):
        """Stop the scheduler"""
        if not self.is_running:
            self.logger.warning("Scheduler is not running")
            return False
            
        try:
            self.stop_event.set()
            schedule.clear()
            self.is_running = False
            self.logger.info("Scheduler stopped successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop scheduler: {str(e)}")
            return False

# API routes
@app.get("/")
async def root():
    """Root endpoint - service health check"""
    return {"status": "ok", "service": "Twitter Poster MCP"}

@app.post("/users/register")
async def register_user(config: UserConfig):
    """Register a new user with the service"""
    # Generate a unique user ID
    user_id = f"user_{secrets.token_hex(8)}"
    
    # Store the API key and configuration
    API_KEYS[config.api_key] = user_id
    USER_CONFIGS[user_id] = config
    
    # Save configuration to disk
    save_config(user_id, config)
    
    logger.info(f"New user registered: {user_id}")
    return {"user_id": user_id, "api_key": config.api_key, "message": "Registration successful"}

@app.post("/twitter/post")
async def create_twitter_post(
    post: PostRequest,
    user_id: str = Depends(get_current_user)
):
    """Create a single Twitter post"""
    if user_id not in USER_CONFIGS:
        raise HTTPException(status_code=404, detail="User configuration not found")
        
    # Create or get TwitterPoster instance
    if user_id not in USER_INSTANCES:
        try:
            USER_INSTANCES[user_id] = TwitterPoster(USER_CONFIGS[user_id].service_config, user_id)
        except Exception as e:
            logger.error(f"Failed to create TwitterPoster instance for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize Twitter poster: {str(e)}")
    
    poster = USER_INSTANCES[user_id]
    
    # Set hashtags if provided
    if post.hashtags:
        poster.fixed_hashtags = post.hashtags
        
    # Create post
    success = poster.create_post(post.content)
    
    if success:
        return {"status": "success", "message": "Tweet posted successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to post tweet")

@app.post("/telegram/fetch")
async def fetch_telegram_messages(
    user_id: str = Depends(get_current_user)
):
    """Fetch latest messages from Telegram channel"""
    if user_id not in USER_CONFIGS:
        raise HTTPException(status_code=404, detail="User configuration not found")
        
    # Create or get TwitterPoster instance
    if user_id not in USER_INSTANCES:
        try:
            USER_INSTANCES[user_id] = TwitterPoster(USER_CONFIGS[user_id].service_config, user_id)
        except Exception as e:
            logger.error(f"Failed to create TwitterPoster instance for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize Twitter poster: {str(e)}")
    
    poster = USER_INSTANCES[user_id]
    
    # Fetch messages
    messages = poster.get_telegram_messages()
    
    if messages:
        return {"status": "success", "messages": messages}
    else:
        return {"status": "success", "messages": "", "note": "No new messages found"}

@app.post("/generate/content")
async def generate_content(
    user_id: str = Depends(get_current_user)
):
    """Generate content from latest Telegram messages"""
    if user_id not in USER_CONFIGS:
        raise HTTPException(status_code=404, detail="User configuration not found")
        
    # Create or get TwitterPoster instance
    if user_id not in USER_INSTANCES:
        try:
            USER_INSTANCES[user_id] = TwitterPoster(USER_CONFIGS[user_id].service_config, user_id)
        except Exception as e:
            logger.error(f"Failed to create TwitterPoster instance for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize Twitter poster: {str(e)}")
    
    poster = USER_INSTANCES[user_id]
    
    # Generate content
    content = poster.generate_post_content()
    
    if content:
        return {"status": "success", "content": content}
    else:
        raise HTTPException(status_code=500, detail="Failed to generate content")

@app.post("/scheduler/start")
async def start_scheduler(
    schedule_config: ScheduleRequest,
    user_id: str = Depends(get_current_user)
):
    """Start the automatic posting scheduler"""
    if user_id not in USER_CONFIGS:
        raise HTTPException(status_code=404, detail="User configuration not found")
        
    # Create or get TwitterPoster instance
    if user_id not in USER_INSTANCES:
        try:
            USER_INSTANCES[user_id] = TwitterPoster(USER_CONFIGS[user_id].service_config, user_id)
        except Exception as e:
            logger.error(f"Failed to create TwitterPoster instance for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize Twitter poster: {str(e)}")
    
    poster = USER_INSTANCES[user_id]
    
    # Update interval settings
    poster.min_interval = schedule_config.min_interval_seconds
    poster.max_interval = schedule_config.max_interval_seconds
    
    # Start scheduler
    success = poster.start_scheduler(schedule_config.interval_minutes)
    
    if success:
        SCHEDULED_TASKS[user_id] = True
        return {"status": "success", "message": f"Scheduler started with {schedule_config.interval_minutes} minute interval"}
    else:
        raise HTTPException(status_code=500, detail="Failed to start scheduler")

@app.post("/scheduler/stop")
async def stop_scheduler(
    user_id: str = Depends(get_current_user)
):
    """Stop the automatic posting scheduler"""
    if user_id not in USER_INSTANCES:
        raise HTTPException(status_code=404, detail="No active instance found for this user")
    
    poster = USER_INSTANCES[user_id]
    
    # Stop scheduler
    success = poster.stop_scheduler()
    
    if success:
        if user_id in SCHEDULED_TASKS:
            del SCHEDULED_TASKS[user_id]
        return {"status": "success", "message": "Scheduler stopped successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to stop scheduler")

@app.get("/status")
async def get_status(
    user_id: str = Depends(get_current_user)
):
    """Get current service status for user"""
    if user_id not in USER_CONFIGS:
        raise HTTPException(status_code=404, detail="User configuration not found")
    
    is_active = user_id in USER_INSTANCES
    is_scheduled = user_id in SCHEDULED_TASKS
    
    return {
        "status": "success",
        "user_id": user_id,
        "active": is_active,
        "scheduled": is_scheduled,
        "config": USER_CONFIGS[user_id].dict() if user_id in USER_CONFIGS else None
    }

@app.put("/config/update")
async def update_config(
    config: ServiceConfig,
    user_id: str = Depends(get_current_user)
):
    """Update user service configuration"""
    if user_id not in USER_CONFIGS:
        raise HTTPException(status_code=404, detail="User configuration not found")
    
    # Update configuration
    USER_CONFIGS[user_id].service_config = config
    
    # Save to disk
    save_config(user_id, USER_CONFIGS[user_id])
    
    # Restart instance if active
    if user_id in USER_INSTANCES:
        was_running = USER_INSTANCES[user_id].is_running
        
        # Stop current instance if it's running
        if was_running:
            USER_INSTANCES[user_id].stop_scheduler()
        
        # Create new instance with updated config
        try:
            USER_INSTANCES[user_id] = TwitterPoster(config, user_id)
            
            # Restart scheduler if it was running
            if was_running:
                USER_INSTANCES[user_id].start_scheduler()
                
            logger.info(f"Successfully restarted instance for user {user_id} with updated config")
        except Exception as e:
            logger.error(f"Failed to restart instance for user {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to restart instance: {str(e)}")
    
    return {"status": "success", "message": "Configuration updated successfully"}

@app.delete("/users/unregister")
async def unregister_user(
    user_id: str = Depends(get_current_user)
):
    """Unregister a user from the service"""
    if user_id not in USER_CONFIGS:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Stop any running instances
    if user_id in USER_INSTANCES:
        if USER_INSTANCES[user_id].is_running:
            USER_INSTANCES[user_id].stop_scheduler()
        
        # Close Telegram client
        if USER_INSTANCES[user_id].telegram_client:
            USER_INSTANCES[user_id].telegram_client.disconnect()
            
        del USER_INSTANCES[user_id]
    
    # Remove scheduled tasks
    if user_id in SCHEDULED_TASKS:
        del SCHEDULED_TASKS[user_id]
    
    # Get API key
    api_key = None
    for key, value in API_KEYS.items():
        if value == user_id:
            api_key = key
            break
    
    # Remove from API keys
    if api_key:
        del API_KEYS[api_key]
    
    # Remove configuration
    config = USER_CONFIGS[user_id]
    del USER_CONFIGS[user_id]
    
    # Remove configuration file
    try:
        config_path = Path("configs") / f"{user_id}.json"
        if config_path.exists():
            config_path.unlink()
    except Exception as e:
        logger.error(f"Failed to delete configuration file for user {user_id}: {str(e)}")
    
    logger.info(f"User {user_id} unregistered successfully")
    return {"status": "success", "message": "User unregistered successfully"}

@app.get("/docs/swagger")
async def get_swagger_docs():
    """Return swagger documentation URL"""
    return {"status": "success", "docs_url": "/docs"}

@app.get("/docs/redoc")
async def get_redoc_docs():
    """Return redoc documentation URL"""
    return {"status": "success", "docs_url": "/redoc"}

@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    return {
        "status": "success",
        "metrics": {
            "registered_users": len(USER_CONFIGS),
            "active_instances": len(USER_INSTANCES),
            "scheduled_tasks": len(SCHEDULED_TASKS)
        }
    }

@app.on_event("startup")
async def startup_event():
    """Execute on application startup"""
    logger.info("Starting Twitter Poster MCP Service...")
    
    # Load existing configurations
    load_configs()
    
    logger.info(f"Loaded {len(USER_CONFIGS)} user configurations")
    logger.info("Twitter Poster MCP Service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Execute on application shutdown"""
    logger.info("Shutting down Twitter Poster MCP Service...")
    
    # Stop all running instances
    for user_id, instance in USER_INSTANCES.items():
        try:
            if instance.is_running:
                instance.stop_scheduler()
            
            # Disconnect Telegram client
            if instance.telegram_client:
                instance.telegram_client.disconnect()
        except Exception as e:
            logger.error(f"Error stopping instance for user {user_id}: {str(e)}")
    
    logger.info("Twitter Poster MCP Service shut down successfully")

# Entry point for running the app
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Twitter Poster MCP Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable hot reloading")
    
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    
    uvicorn.run(
        "main:app", 
        host=args.host, 
        port=args.port, 
        reload=args.reload,
        log_level="info"
    )