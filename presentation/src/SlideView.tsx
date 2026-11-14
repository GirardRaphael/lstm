import type { Slide } from './slides'

type Props = {
  slide: Slide
  index: number
  total: number
}

export function SlideView({ slide, index, total }: Props) {
  return (
    <article className={`slide slide-${slide.variant ?? 'content'}`} data-slide={slide.id}>
      <div className="slide-inner">
        {slide.kicker ? <p className="slide-kicker">{slide.kicker}</p> : null}
        <h1 className="slide-title">{slide.title}</h1>
        {slide.body ? <p className="slide-body">{slide.body}</p> : null}
        {slide.tables?.map((table) => (
          <figure key={table.caption} className="compare-table">
            <figcaption>{table.caption}</figcaption>
            <table>
              <thead>
                <tr>
                  {table.headers.map((header) => (
                    <th key={header}>{header}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((row) => (
                  <tr key={row.join('|')}>
                    {row.map((cell, ci) => (
                      <td key={`${ci}-${cell}`}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </figure>
        ))}
        {slide.bullets ? (
          <ul className="slide-bullets">
            {slide.bullets.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        ) : null}
        {slide.metricRows ? (
          <div className="metric-grid">
            {slide.metricRows.map((row) => (
              <div key={row.label} className="metric-card">
                <div className="metric-value">{row.value}</div>
                <div className="metric-label">{row.label}</div>
                {row.note ? <div className="metric-note">{row.note}</div> : null}
              </div>
            ))}
          </div>
        ) : null}
        {slide.footer ? <p className="slide-footer">{slide.footer}</p> : null}
      </div>
      <div className="slide-index">
        {index + 1} / {total}
      </div>
    </article>
  )
}
