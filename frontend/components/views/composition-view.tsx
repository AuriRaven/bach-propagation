"use client"

// Musical notes for the score visualization
const trebleNotes1 = [
  { x: 15, y: 50 },
  { x: 25, y: 45 },
  { x: 35, y: 55 },
  { x: 45, y: 40 },
  { x: 55, y: 60 },
  { x: 65, y: 35, active: true },
]

const bassNotes1 = [
  { x: 20, y: 45 },
  { x: 30, y: 55 },
  { x: 40, y: 50 },
  { x: 55, y: 45 },
  { x: 65, y: 55 },
]

export function CompositionView() {
  return (
    <div className="bg-card rounded-xl p-8 min-h-full border border-border">
      {/* Title */}
      <h2 className="text-center font-serif text-2xl italic text-foreground/90 mb-8">
        BWV 775
      </h2>

      {/* Musical Score */}
      <div className="space-y-8">
        {/* First system */}
        <div className="relative">
          {/* Treble Clef Line */}
          <div className="flex items-center h-20 border-b border-border/30">
            <div className="w-12 text-2xl text-muted-foreground font-serif">𝄞</div>
            <div className="flex-1 relative h-full">
              {/* Staff lines */}
              {[0, 1, 2, 3, 4].map((i) => (
                <div
                  key={i}
                  className="absolute w-full h-px bg-border/40"
                  style={{ top: `${20 + i * 15}%` }}
                />
              ))}
              {/* Notes */}
              {trebleNotes1.map((note, i) => (
                <div
                  key={i}
                  className={`absolute w-3 h-3 rounded-full ${
                    note.active ? "bg-primary ring-2 ring-primary/50" : "bg-foreground/70"
                  }`}
                  style={{ left: `${note.x}%`, top: `${note.y}%` }}
                />
              ))}
            </div>
          </div>

          {/* Bass Clef Line */}
          <div className="flex items-center h-20 border-b border-border/30">
            <div className="w-12 text-2xl text-muted-foreground font-serif">𝄢</div>
            <div className="flex-1 relative h-full">
              {/* Staff lines */}
              {[0, 1, 2, 3, 4].map((i) => (
                <div
                  key={i}
                  className="absolute w-full h-px bg-border/40"
                  style={{ top: `${20 + i * 15}%` }}
                />
              ))}
              {/* Notes */}
              {bassNotes1.map((note, i) => (
                <div
                  key={i}
                  className="absolute w-3 h-3 rounded-full bg-foreground/70"
                  style={{ left: `${note.x}%`, top: `${note.y}%` }}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Second system (empty for continuation) */}
        <div className="relative opacity-50">
          <div className="flex items-center h-20 border-b border-border/30">
            <div className="w-12 text-2xl text-muted-foreground font-serif">𝄞</div>
            <div className="flex-1 relative h-full">
              {[0, 1, 2, 3, 4].map((i) => (
                <div
                  key={i}
                  className="absolute w-full h-px bg-border/40"
                  style={{ top: `${20 + i * 15}%` }}
                />
              ))}
            </div>
          </div>
          <div className="flex items-center h-20 border-b border-border/30">
            <div className="w-12 text-2xl text-muted-foreground font-serif">𝄢</div>
            <div className="flex-1 relative h-full">
              {[0, 1, 2, 3, 4].map((i) => (
                <div
                  key={i}
                  className="absolute w-full h-px bg-border/40"
                  style={{ top: `${20 + i * 15}%` }}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="flex justify-between text-sm text-muted-foreground mt-8 font-serif italic">
        <span>Bach Propagation Engine v2.0</span>
        <span>Sheet No. 1</span>
      </div>
    </div>
  )
}
