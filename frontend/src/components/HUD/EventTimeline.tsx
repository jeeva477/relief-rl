import { useEffect, useRef } from "react";
import type { SimEvent } from "../../services/sim";

export function EventTimeline({ events }: { events: SimEvent[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = 0;
  }, [events.length]);
  return (
    <div className="timeline" ref={ref}>
      {events.length === 0 && <div className="muted">No events yet. Start a mission.</div>}
      {events.map((ev, i) => (
        <div key={`${i}-${ev.step}-${ev.type}`} className={`tl-row ${ev.severity}`}>
          <span className="tl-step">{ev.step != null ? `S${ev.step}` : "·"}</span>
          <span className="tl-type">{ev.type}</span>
          <span className="tl-text">{ev.text}</span>
        </div>
      ))}
    </div>
  );
}