from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from parakeet_service.streaming_vad import StreamingVAD
from parakeet_service.batchworker        import transcription_queue, condition, results
import asyncio
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws")
async def ws_asr(ws: WebSocket):
    await ws.accept()
    vad = StreamingVAD()
    client_connected = True

    async def producer():
        """push chunks into the global transcription queue"""
        nonlocal client_connected
        try:
            while client_connected:
                frame = await ws.receive_bytes()
                for chunk in vad.feed(frame):
                    await transcription_queue.put(chunk)
                    if client_connected:  # Check before sending
                        try:
                            await ws.send_json({"status": "queued"})
                        except Exception as e:
                            logger.warning(f"Failed to send status message: {e}")
                            client_connected = False
                            break
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
            client_connected = False
        except Exception as e:
            logger.error(f"Producer error: {e}")
            client_connected = False

    async def consumer():
        """stream results back as soon as they're ready"""
        nonlocal client_connected
        try:
            while client_connected:
                async with condition:
                    await condition.wait()
                    
                if not client_connected:
                    break
                    
                flushed = []
                for p, txt in list(results.items()):
                    if not client_connected:
                        break
                    try:
                        await ws.send_json({"text": txt})
                        flushed.append(p)
                    except Exception as e:
                        logger.warning(f"Failed to send transcription result: {e}")
                        client_connected = False
                        break
                        
                for p in flushed:
                    results.pop(p, None)
                    
        except Exception as e:
            logger.error(f"Consumer error: {e}")
            client_connected = False

    try:
        await asyncio.gather(producer(), consumer(), return_exceptions=True)
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")
    finally:
        client_connected = False
        logger.info("WebSocket connection closed")
