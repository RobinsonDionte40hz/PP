import { io, Socket } from 'socket.io-client';
import type { WSMessage } from '../types/api';

const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

class WebSocketService {
  private socket: Socket | null = null;
  private maxReconnectAttempts = 5;
  private messageHandlers: Map<string, Set<(message: WSMessage) => void>> = new Map();
  private currentPredictionId: string | null = null;

  connect(predictionId: string): void {
    // If already connected to the same prediction, don't reconnect
    if (this.socket?.connected && this.currentPredictionId === predictionId) {
      console.log('WebSocket already connected to this prediction');
      return;
    }

    // If connected to a different prediction, disconnect first
    if (this.socket?.connected && this.currentPredictionId !== predictionId) {
      console.log('Switching prediction, disconnecting previous connection');
      this.disconnect();
    }

    this.currentPredictionId = predictionId;

    this.socket = io(WS_BASE_URL, {
      path: '/socket.io',  // Default Socket.IO path at root
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: this.maxReconnectAttempts,
    });

    // Join prediction room
    this.socket.on('connect', () => {
      console.log('✅ WebSocket connected, subscribing to:', predictionId);
      this.socket?.emit('subscribe', { prediction_id: predictionId });
    });

    // Handle messages
    this.socket.on('progress_update', (data) => {
      console.log('📊 Received progress_update:', data);
      this.notifyHandlers(predictionId, { type: 'progress', data });
    });

    this.socket.on('metrics_update', (data) => {
      console.log('📈 Received metrics_update:', data);
      this.notifyHandlers(predictionId, { type: 'metrics', data });
    });

    this.socket.on('agent_update', (data) => {
      console.log('🤖 Received agent_update:', data);
      this.notifyHandlers(predictionId, { type: 'agent', data });
    });

    this.socket.on('event_log', (data) => {
      console.log('📝 Received event_log:', data);
      this.notifyHandlers(predictionId, { type: 'log', data });
    });

    this.socket.on('status_change', (data) => {
      console.log('🔄 Received status_change:', data);
      this.notifyHandlers(predictionId, { type: 'status', data });
    });

    this.socket.on('prediction_complete', (data) => {
      console.log('✅ Received prediction_complete:', data);
      this.notifyHandlers(predictionId, { type: 'complete', data });
    });

    this.socket.on('prediction_error', (data) => {
      console.error('❌ Received prediction_error:', data);
      this.notifyHandlers(predictionId, { type: 'error', data });
    });

    this.socket.on('subscribed', (data) => {
      console.log('Successfully subscribed to prediction:', data.prediction_id);
    });

    // Handle disconnection
    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      if (reason === 'io server disconnect') {
        // Server initiated disconnect, try to reconnect
        this.socket?.connect();
      }
    });

    // Handle reconnection
    this.socket.on('reconnect', (attemptNumber) => {
      console.log('WebSocket reconnected after', attemptNumber, 'attempts');
      this.socket?.emit('subscribe', { prediction_id: predictionId });
    });

    // Handle reconnection error
    this.socket.on('reconnect_failed', () => {
      console.error('WebSocket reconnection failed after', this.maxReconnectAttempts, 'attempts');
    });

    // Handle errors
    this.socket.on('error', (error) => {
      console.error('WebSocket error:', error);
    });
  }

  disconnect(): void {
    if (this.socket) {
      console.log('Disconnecting WebSocket');
      this.socket.disconnect();
      this.socket = null;
      this.currentPredictionId = null;
      this.messageHandlers.clear();
    }
  }

  // Subscribe to messages for a prediction
  subscribe(predictionId: string, handler: (message: WSMessage) => void): () => void {
    if (!this.messageHandlers.has(predictionId)) {
      this.messageHandlers.set(predictionId, new Set());
    }
    
    this.messageHandlers.get(predictionId)!.add(handler);

    // Return unsubscribe function
    return () => {
      const handlers = this.messageHandlers.get(predictionId);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          this.messageHandlers.delete(predictionId);
        }
      }
    };
  }

  // Notify all handlers for a prediction
  private notifyHandlers(predictionId: string, message: WSMessage): void {
    const handlers = this.messageHandlers.get(predictionId);
    if (handlers) {
      handlers.forEach((handler) => handler(message));
    }
  }

  // Check if connected
  isConnected(): boolean {
    return this.socket?.connected || false;
  }

  // Send heartbeat
  sendHeartbeat(): void {
    if (this.socket?.connected) {
      this.socket.emit('heartbeat');
    }
  }
}

// Export singleton instance
export const websocketService = new WebSocketService();
