import { BEST_DICE, fmtDice } from "../lib/metrics";
import { LABELS } from "../lib/labels";

const ROWS = [
  { key: "foreground", name: "Foreground (average of the three)", value: BEST_DICE.foreground, digits: 4, color: "var(--accent)" },
  ...LABELS.map((l) => ({ key: l.key, name: l.name, value: BEST_DICE[l.key], digits: 3, color: l.color })),
] as const;

export default function DiceTable() {
  return (
    <div className="ks-dice" role="table" aria-label="Best Dice scores so far">
      {ROWS.map((r) => (
        <div className="ks-dice-row" role="row" key={r.key}>
          <span className="ks-dice-name" role="rowheader">
            {r.name}
          </span>
          <span className="ks-dice-bar" role="cell" aria-hidden>
            <i style={{ width: `${r.value * 100}%`, background: r.color }} />
          </span>
          <b className="ks-dice-val" role="cell">
            {fmtDice(r.value, r.digits)}
          </b>
        </div>
      ))}
    </div>
  );
}
