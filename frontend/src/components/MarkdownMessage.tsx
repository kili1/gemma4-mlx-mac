import React from "react";
import { Brain, Check, Copy } from "lucide-react";

import { splitThinkingContent } from "../format";

type Block =
  | { kind: "code"; lang: string; content: string }
  | { kind: "text"; content: string };

function splitCodeBlocks(markdown: string): Block[] {
  const blocks: Block[] = [];
  const fenceRe = /```([a-zA-Z0-9_+#.-]*)\n([\s\S]*?)```/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = fenceRe.exec(markdown)) !== null) {
    if (match.index > lastIndex) {
      blocks.push({ kind: "text", content: markdown.slice(lastIndex, match.index) });
    }
    blocks.push({
      kind: "code",
      lang: match[1] || "text",
      content: match[2].replace(/\n$/, ""),
    });
    lastIndex = fenceRe.lastIndex;
  }

  if (lastIndex < markdown.length) {
    blocks.push({ kind: "text", content: markdown.slice(lastIndex) });
  }
  return blocks;
}

function isSafeHref(url: string): boolean {
  return /^https?:\/\//i.test(url.trim());
}

function renderInline(text: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  const tokenRe = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\n]+\*|\[[^\]]+\]\([^)]+\))/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let key = 0;

  while ((match = tokenRe.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }
    const token = match[0];
    if (token.startsWith("`")) {
      nodes.push(<code key={key++}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith("**")) {
      nodes.push(<strong key={key++}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith("[")) {
      const linkMatch = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(token);
      if (linkMatch && isSafeHref(linkMatch[2])) {
        nodes.push(
          <a key={key++} href={linkMatch[2]} target="_blank" rel="noopener noreferrer">
            {linkMatch[1]}
          </a>,
        );
      } else {
        nodes.push(token);
      }
    } else {
      nodes.push(<em key={key++}>{token.slice(1, -1)}</em>);
    }
    lastIndex = tokenRe.lastIndex;
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }
  return nodes;
}

function parseTable(lines: string[]): { headers: string[]; rows: string[][] } | null {
  if (lines.length < 2) {
    return null;
  }
  const trimmed = lines.map((line) => line.trim());
  if (!trimmed[0].includes("|") || !/^\|?[\s:-]+\|[\s|:-]*$/.test(trimmed[1])) {
    return null;
  }
  const splitRow = (line: string) =>
    line
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => cell.trim());

  const headers = splitRow(trimmed[0]);
  const rows = trimmed.slice(2).map(splitRow);
  return { headers, rows };
}

function CodeBlock({ lang, content }: { lang: string; content: string }) {
  const [copied, setCopied] = React.useState(false);

  async function copyCode() {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      // Clipboard API may be unavailable in non-secure contexts.
    }
  }

  return (
    <div className="md-code-block">
      <div className="md-code-header">
        <span>{lang}</span>
        <button type="button" className="copy-button" onClick={copyCode} title="Copy code">
          {copied ? <Check size={13} /> : <Copy size={13} />}
          <span>{copied ? "Copied" : "Copy"}</span>
        </button>
      </div>
      <pre>
        <code>{content}</code>
      </pre>
    </div>
  );
}

function MarkdownText({ content }: { content: string }) {
  const blocks = splitCodeBlocks(content);
  return (
    <div className="md-body">
      {blocks.map((block, blockIdx) => {
        if (block.kind === "code") {
          return <CodeBlock key={blockIdx} lang={block.lang} content={block.content} />;
        }

        const paragraphs = block.content
          .split(/\n{2,}/)
          .filter((chunk) => chunk.trim().length > 0);
        return (
          <React.Fragment key={blockIdx}>
            {paragraphs.map((paragraph, paraIdx) => {
              const lines = paragraph.split("\n");
              const singleTrimmed = paragraph.trim();

              if (/^(-{3,}|\*{3,}|_{3,})$/.test(singleTrimmed)) {
                return <hr key={paraIdx} className="md-hr" />;
              }

              const table = parseTable(lines);
              if (table) {
                return (
                  <div className="md-table-wrap" key={paraIdx}>
                    <table className="md-table">
                      <thead>
                        <tr>
                          {table.headers.map((head, hIdx) => (
                            <th key={hIdx}>{renderInline(head)}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {table.rows.map((row, rIdx) => (
                          <tr key={rIdx}>
                            {row.map((cell, cIdx) => (
                              <td key={cIdx}>{renderInline(cell)}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                );
              }

              if (lines.every((line) => /^\s*>\s?/.test(line))) {
                return (
                  <blockquote key={paraIdx} className="md-quote">
                    {lines.map((line, qIdx) => (
                      <React.Fragment key={qIdx}>
                        {qIdx > 0 && <br />}
                        {renderInline(line.replace(/^\s*>\s?/, ""))}
                      </React.Fragment>
                    ))}
                  </blockquote>
                );
              }

              if (lines.every((line) => /^\s*[-*]\s+/.test(line))) {
                return (
                  <ul key={paraIdx}>
                    {lines.map((line, liIdx) => (
                      <li key={liIdx}>{renderInline(line.replace(/^\s*[-*]\s+/, ""))}</li>
                    ))}
                  </ul>
                );
              }
              if (lines.every((line) => /^\s*\d+\.\s+/.test(line))) {
                return (
                  <ol key={paraIdx}>
                    {lines.map((line, liIdx) => (
                      <li key={liIdx}>{renderInline(line.replace(/^\s*\d+\.\s+/, ""))}</li>
                    ))}
                  </ol>
                );
              }

              const heading = /^(#{1,4})\s+(.+)$/.exec(singleTrimmed);
              if (heading && lines.length === 1) {
                const level = heading[1].length;
                const HeadingTag = level <= 2 ? "h3" : "h4";
                return <HeadingTag key={paraIdx}>{renderInline(heading[2])}</HeadingTag>;
              }

              return (
                <p key={paraIdx}>
                  {lines.map((line, lineIdx) => (
                    <React.Fragment key={lineIdx}>
                      {lineIdx > 0 && <br />}
                      {renderInline(line)}
                    </React.Fragment>
                  ))}
                </p>
              );
            })}
          </React.Fragment>
        );
      })}
    </div>
  );
}

export function MarkdownMessage({
  content,
  streaming = false,
}: {
  content: string;
  streaming?: boolean;
}) {
  const { thinking, answer } = splitThinkingContent(content);
  const thinkingWords = thinking ? thinking.trim().split(/\s+/).filter(Boolean).length : 0;

  return (
    <div className="assistant-body">
      {thinking !== null && (
        <details className="thinking-block" open={streaming && !answer}>
          <summary>
            <Brain size={14} className={streaming && !answer ? "thinking-pulse" : ""} />
            <span>{streaming && !answer ? "Thinking…" : "Thought process"}</span>
            {thinkingWords > 0 && <small className="thinking-count">{thinkingWords} words</small>}
          </summary>
          <pre className="thinking-trace">{thinking}</pre>
        </details>
      )}
      {answer ? (
        <MarkdownText content={answer} />
      ) : thinking !== null && streaming ? (
        <p className="pending-copy">Drafting answer…</p>
      ) : null}
    </div>
  );
}
