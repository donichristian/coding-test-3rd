"""
Chat API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
import uuid
from datetime import datetime
from app.db.session import get_db
from app.core.celery_app import celery_app
from app.schemas.chat import (
    ChatQueryRequest,
    ChatQueryResponse,
    ConversationCreate,
    Conversation,
    ChatMessage
)

router = APIRouter()

# In-memory conversation storage (replace with Redis/DB in production)
conversations: Dict[str, Dict[str, Any]] = {}


@router.post("/query", response_model=ChatQueryResponse)
async def process_chat_query(
    request: ChatQueryRequest,
    db: Session = Depends(get_db)
):
    """Process a chat query using RAG via Celery task"""

    # Check if any documents exist in the system
    from app.models.document import Document
    total_docs = db.query(Document).filter(Document.parsing_status == "completed").count()

    if total_docs == 0:
        # No completed documents in the entire system
        return ChatQueryResponse(
            answer="No documents have been uploaded and processed yet. Please upload a fund performance report first, then I can help you analyze metrics like DPI, IRR, and answer questions about capital calls, distributions, and adjustments.",
            sources=[],
            metrics=None,
            processing_time=0.0,
            task_id=""
        )

    # Determine document_id from fund_id if needed
    document_id = None
    if request.fund_id:
        # Find the most recent completed document for this fund
        from app.models.document import Document
        recent_doc = db.query(Document).filter(
            Document.fund_id == request.fund_id,
            Document.parsing_status == "completed"
        ).order_by(Document.upload_date.desc()).first()

        if recent_doc:
            document_id = recent_doc.id
        else:
            # Check if fund has any documents at all (even failed ones)
            any_docs = db.query(Document).filter(Document.fund_id == request.fund_id).count()
            if any_docs == 0:
                # No documents uploaded for this fund yet
                return ChatQueryResponse(
                    answer="No documents have been uploaded for this fund yet. Please upload a fund performance report first, then I can help you analyze metrics and answer questions about capital calls, distributions, and adjustments.",
                    sources=[],
                    metrics=None,
                    processing_time=0.0,
                    task_id=""
                )

    # Submit chat processing task to Celery
    task = celery_app.send_task(
        'app.tasks.chat_tasks.process_chat_query_task',
        args=[
            request.query,
            request.fund_id,
            request.conversation_id,
            document_id
        ]
    )

    # Return task_id for frontend polling - no content response
    return ChatQueryResponse(
        answer="",
        sources=[],
        metrics=None,
        processing_time=0.0,
        task_id=task.id
    )


@router.post("/conversations", response_model=Conversation)
async def create_conversation(request: ConversationCreate):
    """Create a new conversation"""
    conversation_id = str(uuid.uuid4())
    
    conversations[conversation_id] = {
        "fund_id": request.fund_id,
        "messages": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    return Conversation(
        conversation_id=conversation_id,
        fund_id=request.fund_id,
        messages=[],
        created_at=conversations[conversation_id]["created_at"],
        updated_at=conversations[conversation_id]["updated_at"]
    )


@router.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conv = conversations[conversation_id]
    
    return Conversation(
        conversation_id=conversation_id,
        fund_id=conv["fund_id"],
        messages=[ChatMessage(**msg) for msg in conv["messages"]],
        created_at=conv["created_at"],
        updated_at=conv["updated_at"]
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    del conversations[conversation_id]

    return {"message": "Conversation deleted successfully"}


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get the status and result of a chat processing task"""
    try:
        from app.core.celery_app import celery_app
        task_result = celery_app.AsyncResult(task_id)

        if task_result.state == "PENDING":
            return {
                "task_id": task_id,
                "status": "pending",
                "message": "Task is still processing"
            }
        elif task_result.state == "PROGRESS":
            return {
                "task_id": task_id,
                "status": "progress",
                "message": task_result.info.get("message", "Processing...")
            }
        elif task_result.state == "SUCCESS":
            result = task_result.result
            return {
                "task_id": task_id,
                "status": "completed",
                "result": result
            }
        else:
            # FAILURE or other states
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(task_result.info) if task_result.info else "Unknown error"
            }
    except Exception as e:
        return {
            "task_id": task_id,
            "status": "error",
            "error": f"Failed to get task status: {str(e)}"
        }
