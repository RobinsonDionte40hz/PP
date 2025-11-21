import { useEffect, useState, useCallback, useRef } from 'react';
import { websocketService } from '../services/websocketService';
import type { WSMessage, PredictionProgress } from '../types/api';

// Configuration for performance optimization
const MESSAGE_BATCH_INTERVAL = 100; // Batch messages every 100ms
const PROGRESS_THROTTLE_INTERVAL = 250; // Update progress UI every 250ms
const MAX_MESSAGE_HISTORY = 100; // Keep only last 100 messages in memory

export function useWebSocket(predictionId: string | undefined) {
  const [isConnected, setIsConnected] = useState(false);
  const [latestProgress, setLatestProgress] = useState<PredictionProgress | null>(null);
  const [messages, setMessages] = useState<WSMessage[]>([]);

  // Refs for batching and throttling
  const messageBatchRef = useRef<WSMessage[]>([]);
  const messageBatchTimerRef = useRef<NodeJS.Timeout | null>(null);
  const lastProgressUpdateRef = useRef<number>(0);
  const pendingProgressRef = useRef<PredictionProgress | null>(null);

  // Flush message batch to state
  const flushMessageBatch = useCallback(() => {
    if (messageBatchRef.current.length > 0) {
      setMessages((prev) => {
        const newMessages = [...prev, ...messageBatchRef.current];
        // Keep only last MAX_MESSAGE_HISTORY messages to prevent memory issues
        return newMessages.slice(-MAX_MESSAGE_HISTORY);
      });
      messageBatchRef.current = [];
    }
  }, []);

  // Throttled progress update
  const updateProgress = useCallback((progress: PredictionProgress) => {
    const now = Date.now();
    const timeSinceLastUpdate = now - lastProgressUpdateRef.current;

    if (timeSinceLastUpdate >= PROGRESS_THROTTLE_INTERVAL) {
      // Update immediately
      console.log('📊 Setting latestProgress (throttled):', progress);
      setLatestProgress(progress);
      lastProgressUpdateRef.current = now;
      pendingProgressRef.current = null;
    } else {
      // Store pending update
      pendingProgressRef.current = progress;
    }
  }, []);

  // Handle incoming message with batching and throttling
  const handleMessage = useCallback((message: WSMessage) => {
    console.log('📬 useWebSocket received message:', message);
    
    // Add to batch
    messageBatchRef.current.push(message);
    
    // Handle progress messages with throttling
    if (message.type === 'progress') {
      updateProgress(message.data);
    }
    
    // Schedule batch flush if not already scheduled
    if (!messageBatchTimerRef.current) {
      messageBatchTimerRef.current = setTimeout(() => {
        flushMessageBatch();
        messageBatchTimerRef.current = null;
      }, MESSAGE_BATCH_INTERVAL);
    }
  }, [flushMessageBatch, updateProgress]);

  // Effect to handle pending progress updates
  useEffect(() => {
    const progressTimer = setInterval(() => {
      if (pendingProgressRef.current) {
        console.log('📊 Setting latestProgress (pending):', pendingProgressRef.current);
        setLatestProgress(pendingProgressRef.current);
        lastProgressUpdateRef.current = Date.now();
        pendingProgressRef.current = null;
      }
    }, PROGRESS_THROTTLE_INTERVAL);

    return () => clearInterval(progressTimer);
  }, []);

  useEffect(() => {
    if (!predictionId) return;

    // Connect to WebSocket
    websocketService.connect(predictionId);
    setIsConnected(websocketService.isConnected());

    // Subscribe to messages
    const unsubscribe = websocketService.subscribe(predictionId, handleMessage);

    // Check connection status
    const checkConnection = setInterval(() => {
      setIsConnected(websocketService.isConnected());
    }, 1000);

    // Cleanup
    return () => {
      clearInterval(checkConnection);
      unsubscribe();
      
      // Flush any remaining messages
      if (messageBatchTimerRef.current) {
        clearTimeout(messageBatchTimerRef.current);
        messageBatchTimerRef.current = null;
      }
      flushMessageBatch();
      
      // Don't disconnect here - let the component unmount fully disconnect
    };
  }, [predictionId, handleMessage, flushMessageBatch]);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return {
    isConnected,
    latestProgress,
    messages,
    clearMessages,
  };
}
