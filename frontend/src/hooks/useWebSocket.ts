import { useEffect, useState, useCallback } from 'react';
import { websocketService } from '../services/websocketService';
import type { WSMessage, PredictionProgress } from '../types/api';

export function useWebSocket(predictionId: string | undefined) {
  const [isConnected, setIsConnected] = useState(false);
  const [latestProgress, setLatestProgress] = useState<PredictionProgress | null>(null);
  const [messages, setMessages] = useState<WSMessage[]>([]);

  const handleMessage = useCallback((message: WSMessage) => {
    setMessages((prev) => [...prev, message]);
    
    if (message.type === 'progress') {
      setLatestProgress(message.data);
    }
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
      // Don't disconnect here - let the component unmount fully disconnect
    };
  }, [predictionId, handleMessage]);

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
