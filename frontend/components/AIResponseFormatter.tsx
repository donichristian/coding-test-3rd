import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface AIResponseFormatterProps {
  content: string;
  className?: string;
}

export const AIResponseFormatter: React.FC<AIResponseFormatterProps> = ({
  content,
  className = ''
}) => {
  // Custom components for better styling
  const components = {
    // Enhanced code blocks with copy functionality
    code: ({ node, inline, className, children, ...props }: any) => {
      if (inline) {
        return (
          <code
            className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono"
            {...props}
          >
            {children}
          </code>
        );
      }

      return (
        <div className="relative group">
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <code className={className} {...props}>
              {children}
            </code>
          </pre>
          <button
            className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded text-xs"
            onClick={() => navigator.clipboard.writeText(String(children))}
          >
            Copy
          </button>
        </div>
      );
    },

    // Enhanced blockquotes
    blockquote: ({ children }: any) => (
      <blockquote className="border-l-4 border-blue-500 pl-4 py-2 my-4 bg-blue-50 italic">
        {children}
      </blockquote>
    ),

    // Enhanced lists
    ul: ({ children }: any) => (
      <ul className="list-disc list-inside space-y-1 my-4">
        {children}
      </ul>
    ),

    ol: ({ children }: any) => (
      <ol className="list-decimal list-inside space-y-1 my-4">
        {children}
      </ol>
    ),

    // Enhanced headings
    h1: ({ children }: any) => (
      <h1 className="text-2xl font-bold mb-4 mt-6 text-gray-900">
        {children}
      </h1>
    ),

    h2: ({ children }: any) => (
      <h2 className="text-xl font-semibold mb-3 mt-5 text-gray-800">
        {children}
      </h2>
    ),

    h3: ({ children }: any) => (
      <h3 className="text-lg font-medium mb-2 mt-4 text-gray-700">
        {children}
      </h3>
    ),

    // Enhanced paragraphs
    p: ({ children }: any) => (
      <p className="mb-4 leading-relaxed text-gray-700">
        {children}
      </p>
    ),

    // Enhanced links
    a: ({ children, href }: any) => (
      <a
        href={href}
        className="text-blue-600 hover:text-blue-800 underline"
        target="_blank"
        rel="noopener noreferrer"
      >
        {children}
      </a>
    ),

    // Enhanced tables
    table: ({ children }: any) => (
      <div className="overflow-x-auto my-4">
        <table className="min-w-full border-collapse border border-gray-300">
          {children}
        </table>
      </div>
    ),

    th: ({ children }: any) => (
      <th className="border border-gray-300 px-4 py-2 bg-gray-100 font-semibold text-left">
        {children}
      </th>
    ),

    td: ({ children }: any) => (
      <td className="border border-gray-300 px-4 py-2">
        {children}
      </td>
    ),

    // Enhanced horizontal rules
    hr: () => (
      <hr className="my-8 border-gray-300" />
    ),

    // Enhanced emphasis
    strong: ({ children }: any) => (
      <strong className="font-semibold text-gray-900">
        {children}
      </strong>
    ),

    em: ({ children }: any) => (
      <em className="italic text-gray-700">
        {children}
      </em>
    ),
  };

  return (
    <div className={`prose prose-sm max-w-none ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default AIResponseFormatter;