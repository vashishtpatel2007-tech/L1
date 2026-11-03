import { useEffect, useRef, useState } from "react";

export type WSMessage = { kind: string; data: any };

/** Auto-reconnecting websocket. Reconnect matters: a dropped socket mid-demo
 *  must heal itself without anyone touching the keyboard. */
export function useWS(onMessage: (m: WSMessage) => void) {
  const [connected, setConnected] = useState(false);
  const cb = useRef(onMessage);
  cb.current = onMessage;

  useEffect(() => {
    let ws: WebSocket | null = null;
    let timer: number | undefined;
    let closed = false;

    const connect = () => {
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      const url = import.meta.env.DEV
        ? `${proto}//${location.host}/ws`
        : `${proto}//${location.host}/ws`;
      ws = new WebSocket(url);
      ws.onopen = () => setConnected(true);
      ws.onmessage = (e) => {
        try { cb.current(JSON.parse(e.data)); } catch { /* ignore */ }
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) timer = window.setTimeout(connect, 1500);
      };
      ws.onerror = () => ws?.close();
    };
    connect();

    const ping = window.setInterval(() => {
      if (ws?.readyState === WebSocket.OPEN) ws.send("ping");
    }, 20000);

    return () => {
      closed = true;
      window.clearTimeout(timer);
      window.clearInterval(ping);
      ws?.close();
    };
  }, []);

  return connected;
}
