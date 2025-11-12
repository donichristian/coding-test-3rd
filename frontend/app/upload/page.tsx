"use client";

import { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, CheckCircle, XCircle, Loader2, Table, FileText } from "lucide-react";
import { documentApi, fundApi } from "@/lib/api";
import { formatCurrency } from "@/lib/utils";

const UPLOAD_STATUS_KEY = 'upload_status';

export default function UploadPage() {
   const [uploading, setUploading] = useState(false);
   const [uploadStatus, setUploadStatus] = useState<{
     status: "idle" | "uploading" | "processing" | "success" | "error";
     message?: string;
     documentId?: number;
     progress?: number;
     startTime?: number;
     elapsedTime?: number;
     processingResult?: any;
     extractedData?: {
       capitalCalls: any[];
       distributions: any[];
       adjustments: any[];
       metrics?: any;
     };
     retryCount?: number;
     maxRetries?: number;
   }>({ status: "idle" });

  // Load persisted upload status on component mount
  useEffect(() => {
    const persistedStatus = localStorage.getItem(UPLOAD_STATUS_KEY);
    if (persistedStatus) {
      try {
        const parsedStatus = JSON.parse(persistedStatus);
        setUploadStatus(parsedStatus);
        setUploading(parsedStatus.status === "uploading" || parsedStatus.status === "processing");
      } catch (error) {
        console.error('Failed to parse persisted upload status:', error);
        localStorage.removeItem(UPLOAD_STATUS_KEY);
      }
    }
  }, []);

  // Persist upload status to localStorage whenever it changes
  useEffect(() => {
    if (uploadStatus.status !== "idle") {
      localStorage.setItem(UPLOAD_STATUS_KEY, JSON.stringify(uploadStatus));
    } else {
      localStorage.removeItem(UPLOAD_STATUS_KEY);
    }
  }, [uploadStatus]);

  const clearPersistedData = useCallback(() => {
    localStorage.removeItem(UPLOAD_STATUS_KEY);
  }, []);

  // Update elapsed time every second during processing
  useEffect(() => {
    const interval = setInterval(() => {
      setUploadStatus((prev) => {
        if (prev.status === "processing" && prev.startTime) {
          return {
            ...prev,
            elapsedTime: Math.floor((Date.now() - prev.startTime) / 1000),
          };
        }
        return prev;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const onDrop = useCallback(async (acceptedFiles: File[], retryCount = 0) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    const maxRetries = 3;

    // Clear any previous persisted data when starting new upload
    clearPersistedData();

    setUploading(true);
    setUploadStatus({
      status: "uploading",
      message: retryCount > 0 ? `Retrying upload... (attempt ${retryCount + 1}/${maxRetries + 1})` : "Uploading file...",
      retryCount,
      maxRetries
    });

    try {
      const result = await documentApi.upload(file);

      setUploadStatus({
        status: "processing",
        message: "File uploaded. Processing document...",
        documentId: result.document_id,
        progress: 10,
        startTime: Date.now(),
        retryCount,
        maxRetries
      });

      // Poll for status
      pollDocumentStatus(result.document_id, retryCount);
    } catch (error: any) {
      const isNetworkError = !error.response || error.code === 'NETWORK_ERROR' || error.code === 'ECONNABORTED';
      const isRetryableError = isNetworkError || error.response?.status >= 500;

      if (isRetryableError && retryCount < maxRetries) {
        const delay = Math.min(1000 * Math.pow(2, retryCount), 10000); // Exponential backoff, max 10s
        console.log(`Upload failed, retrying in ${delay}ms... (attempt ${retryCount + 1}/${maxRetries + 1})`);

        setUploadStatus({
          status: "uploading",
          message: `Upload failed, retrying in ${Math.ceil(delay / 1000)}s... (attempt ${retryCount + 1}/${maxRetries + 1})`,
          retryCount: retryCount + 1,
          maxRetries
        });

        setTimeout(() => {
          onDrop(acceptedFiles, retryCount + 1);
        }, delay);
      } else {
        setUploadStatus({
          status: "error",
          message: retryCount > 0
            ? `Upload failed after ${retryCount + 1} attempts: ${error.response?.data?.detail || error.message || "Unknown error"}`
            : error.response?.data?.detail || "Upload failed",
          retryCount,
          maxRetries
        });
        setUploading(false);
      }
    }
  }, []);

  const pollDocumentStatus = async (documentId: number, retryCount = 0) => {
    const maxAttempts = 60; // 5 minutes max (reduced from 10 minutes)
    let attempts = 0;

    const poll = async () => {
      try {
        const status = await documentApi.getStatus(documentId);

        if (status.status === "completed") {
          // Fetch processing results and extracted data
          try {
            const documentDetails = await documentApi.getDetails(documentId);

            // Fetch extracted data from the fund (assuming fund_id is available)
            let extractedData = null;
            if (documentDetails.fund_id) {
              try {
                console.log('Fetching data for fund_id:', documentDetails.fund_id);

                const fundData = await fundApi.get(documentDetails.fund_id);
                console.log('Fund data:', fundData);

                const capitalCalls = await fundApi.getTransactions(documentDetails.fund_id, 'capital_calls');
                console.log('Capital calls:', capitalCalls);

                const distributions = await fundApi.getTransactions(documentDetails.fund_id, 'distributions');
                console.log('Distributions:', distributions);

                const adjustments = await fundApi.getTransactions(documentDetails.fund_id, 'adjustments');
                console.log('Adjustments:', adjustments);

                const metrics = await fundApi.getMetrics(documentDetails.fund_id);
                console.log('Metrics:', metrics);

                extractedData = {
                  capitalCalls: capitalCalls.items || [],
                  distributions: distributions.items || [],
                  adjustments: adjustments.items || [],
                  metrics
                };

                console.log('Final extractedData:', extractedData);
                console.log('Raw API responses:');
                console.log('Capital calls response:', JSON.stringify(capitalCalls, null, 2));
                console.log('Distributions response:', JSON.stringify(distributions, null, 2));
                console.log('Adjustments response:', JSON.stringify(adjustments, null, 2));
                console.log('Metrics response:', JSON.stringify(metrics, null, 2));
              } catch (dataError) {
                console.error('Could not fetch extracted data:', dataError);
              }
            } else {
              console.log('No fund_id in document details:', documentDetails);
            }

            setUploadStatus({
              status: "success",
              message: "Document processed successfully!",
              documentId,
              progress: 100,
              processingResult: {
                fileName: documentDetails.file_name,
                uploadDate: documentDetails.upload_date,
                parsingStatus: documentDetails.parsing_status,
              },
              extractedData: extractedData || undefined,
              retryCount,
              maxRetries: 3
            });
          } catch (error) {
            setUploadStatus({
              status: "success",
              message: "Document processed successfully!",
              documentId,
              progress: 100,
              retryCount,
              maxRetries: 3
            });
          }
          setUploading(false);
        } else if (status.status === "failed") {
          setUploadStatus({
            status: "error",
            message: status.error_message || "Processing failed",
            documentId,
            retryCount,
            maxRetries: 3
          });
          setUploading(false);
        } else if (attempts < maxAttempts) {
          attempts++;
          // Better progress estimation based on expected processing time
          // Models are pre-loaded, so processing should be faster
          const progress = Math.min(
            90,
            Math.floor((attempts / maxAttempts) * 80) + 10
          );
          setUploadStatus((prev) => ({
            ...prev,
            progress: progress,
          }));
          setTimeout(poll, 5000); // Poll every 5 seconds
        } else {
          setUploadStatus({
            status: "error",
            message:
              "Processing timeout - document processing should complete quickly (models are pre-loaded)",
            documentId,
            retryCount,
            maxRetries: 3
          });
          setUploading(false);
        }
      } catch (error: any) {
        const isNetworkError = !error.response || error.code === 'NETWORK_ERROR' || error.code === 'ECONNABORTED';
        const isRetryableError = isNetworkError || error.response?.status >= 500;

        if (isRetryableError && retryCount < 3) {
          const delay = Math.min(2000 * Math.pow(2, retryCount), 15000); // Exponential backoff, max 15s
          console.log(`Status check failed, retrying in ${delay}ms... (attempt ${retryCount + 1}/4)`);

          setUploadStatus((prev) => ({
            ...prev,
            message: `Connection lost, retrying in ${Math.ceil(delay / 1000)}s... (attempt ${retryCount + 1}/4)`,
          }));

          setTimeout(() => {
            pollDocumentStatus(documentId, retryCount + 1);
          }, delay);
        } else {
          setUploadStatus({
            status: "error",
            message: retryCount > 0
              ? `Failed to check status after ${retryCount + 1} attempts: ${error.message || "Network error"}`
              : "Failed to check status",
            documentId,
            retryCount,
            maxRetries: 3
          });
          setUploading(false);
        }
      }
    };

    poll();
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (acceptedFiles: File[]) => onDrop(acceptedFiles),
    accept: {
      "application/pdf": [".pdf"],
    },
    maxFiles: 1,
    disabled: uploading,
  });

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Upload Fund Document</h1>
        <p className="text-gray-600">
          Upload a PDF fund performance report to automatically extract and
          analyze data
        </p>
      </div>

      {/* Upload Area */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition
          ${
            isDragActive
              ? "border-blue-500 bg-blue-50"
              : "border-gray-300 hover:border-gray-400"
          }
          ${uploading ? "opacity-50 cursor-not-allowed" : ""}
        `}
      >
        <input {...getInputProps()} />

        <div className="flex flex-col items-center">
          <Upload className="w-16 h-16 text-gray-400 mb-4" />

          {isDragActive ? (
            <p className="text-lg text-blue-600 font-medium">
              Drop the file here...
            </p>
          ) : (
            <>
              <p className="text-lg font-medium mb-2">
                Drag & drop a PDF file here, or click to select
              </p>
              <p className="text-sm text-gray-500">Maximum file size: 50MB</p>
            </>
          )}
        </div>
      </div>

      {/* Status Display */}
      {uploadStatus.status !== "idle" && (
        <div className="mt-8">
          <div
            className={`
            rounded-lg p-6 border
            ${
              uploadStatus.status === "success"
                ? "bg-green-50 border-green-200"
                : ""
            }
            ${uploadStatus.status === "error" ? "bg-red-50 border-red-200" : ""}
            ${
              uploadStatus.status === "uploading" ||
              uploadStatus.status === "processing"
                ? "bg-blue-50 border-blue-200"
                : ""
            }
          `}
          >
            <div className="flex items-start">
              <div className="flex-shrink-0">
                {uploadStatus.status === "success" && (
                  <CheckCircle className="w-6 h-6 text-green-600" />
                )}
                {uploadStatus.status === "error" && (
                  <XCircle className="w-6 h-6 text-red-600" />
                )}
                {(uploadStatus.status === "uploading" ||
                  uploadStatus.status === "processing") && (
                  <Loader2 className="w-6 h-6 text-blue-600 animate-spin" />
                )}
              </div>

              <div className="ml-4 flex-1">
                <h3
                  className={`
                  font-medium
                  ${uploadStatus.status === "success" ? "text-green-900" : ""}
                  ${uploadStatus.status === "error" ? "text-red-900" : ""}
                  ${
                    uploadStatus.status === "uploading" ||
                    uploadStatus.status === "processing"
                      ? "text-blue-900"
                      : ""
                  }
                `}
                >
                  {uploadStatus.status === "uploading" && "Uploading..."}
                  {uploadStatus.status === "processing" && "Processing..."}
                  {uploadStatus.status === "success" && "Success!"}
                  {uploadStatus.status === "error" && "Error"}
                </h3>

                <p
                  className={`
                  mt-1 text-sm
                  ${uploadStatus.status === "success" ? "text-green-700" : ""}
                  ${uploadStatus.status === "error" ? "text-red-700" : ""}
                  ${
                    uploadStatus.status === "uploading" ||
                    uploadStatus.status === "processing"
                      ? "text-blue-700"
                      : ""
                  }
                `}
                >
                  {uploadStatus.message}
                </p>

                {uploadStatus.status === "processing" &&
                  uploadStatus.progress !== undefined && (
                    <div className="mt-3">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${uploadStatus.progress}%` }}
                        ></div>
                      </div>
                      <p className="text-xs text-gray-500 mt-1 text-center">
                        {uploadStatus.progress}% complete
                        {uploadStatus.elapsedTime !== undefined && (
                          <span className="block text-xs mt-1">
                            Elapsed: {uploadStatus.elapsedTime}s
                          </span>
                        )}
                        {uploadStatus.retryCount !== undefined && uploadStatus.retryCount > 0 && (
                          <span className="block text-xs mt-1 text-orange-600">
                            Retry: {uploadStatus.retryCount}/{uploadStatus.maxRetries || 3}
                          </span>
                        )}
                      </p>
                      <div className="mt-3 flex justify-center gap-2">
                        <button
                          onClick={() => {
                            setUploadStatus({ status: "idle" });
                            setUploading(false);
                            clearPersistedData();
                          }}
                          className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition text-sm"
                        >
                          Cancel Processing
                        </button>
                        {uploadStatus.retryCount !== undefined && uploadStatus.retryCount > 0 && (
                          <button
                            onClick={() => {
                              // Retry the current operation
                              if (uploadStatus.documentId) {
                                pollDocumentStatus(uploadStatus.documentId, uploadStatus.retryCount);
                              }
                            }}
                            className="px-4 py-2 bg-orange-600 text-white rounded-md hover:bg-orange-700 transition text-sm"
                          >
                            Retry Now
                          </button>
                        )}
                      </div>
                    </div>
                  )}

                {uploadStatus.status === "success" && (
                  <div className="mt-4 space-y-6">
                    {/* Processing Results */}
                    {uploadStatus.processingResult && (
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-medium text-gray-900 mb-2">Processing Results</h4>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-gray-600">File:</span>
                            <span className="ml-2 font-medium">{uploadStatus.processingResult.fileName}</span>
                          </div>
                          <div>
                            <span className="text-gray-600">Status:</span>
                            <span className="ml-2 font-medium text-green-600">✓ Completed</span>
                          </div>
                          <div>
                            <span className="text-gray-600">Uploaded:</span>
                            <span className="ml-2 font-medium">
                              {uploadStatus.processingResult.uploadDate ?
                                new Date(uploadStatus.processingResult.uploadDate).toLocaleString() :
                                'Just now'
                              }
                            </span>
                          </div>
                          <div>
                            <span className="text-gray-600">Processing:</span>
                            <span className="ml-2 font-medium text-green-600">✓ Successful</span>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Extracted Data Display */}
                    {uploadStatus.extractedData && (
                      <div className="space-y-4">
                        {/* Processing Method & Metrics Summary */}
                        {uploadStatus.extractedData.metrics && (
                          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                            <div className="flex items-center justify-between mb-3">
                              <div className="flex items-center justify-between w-full">
                                <div className="flex items-center">
                                  <h4 className="font-medium text-blue-900 flex items-center">
                                    <FileText className="w-4 h-4 mr-2" />
                                    Calculated Metrics
                                  </h4>
                                  {(uploadStatus.extractedData as any).processingMethod && (
                                    <span className="ml-2 text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                                      {(uploadStatus.extractedData as any).processingMethod === 'docling' ? '🤖 Docling' : '📄 pdfplumber'}
                                    </span>
                                  )}
                                </div>
                              </div>
                            </div>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                              {Object.entries(uploadStatus.extractedData.metrics).map(([key, value]) => {
                                if (value === null || value === undefined) return null;

                                let displayValue: string;
                                if (typeof value === 'number' && key.toLowerCase().includes('irr')) {
                                  displayValue = `${value.toFixed(2)}%`;
                                } else if (typeof value === 'number' && key.toLowerCase() === 'dpi') {
                                  // DPI is a ratio (0.3598), display as percentage (35.98%)
                                  displayValue = `${(value * 100).toFixed(2)}%`;
                                } else if (typeof value === 'number') {
                                  displayValue = formatCurrency(value);
                                } else {
                                  displayValue = String(value);
                                }

                                return (
                                  <div key={key} className="bg-white rounded p-2 text-center">
                                    <div className="text-xs text-gray-600 uppercase">{key}</div>
                                    <div className="text-sm font-semibold text-blue-900">{displayValue}</div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}

                        {/* Capital Calls Table */}
                        {uploadStatus.extractedData.capitalCalls && uploadStatus.extractedData.capitalCalls.length > 0 && (
                          <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
                            <div className="bg-gray-50 px-4 py-3 border-b border-gray-200">
                              <h4 className="font-medium text-gray-900 flex items-center">
                                <Table className="w-4 h-4 mr-2" />
                                Capital Calls ({uploadStatus.extractedData.capitalCalls.length})
                              </h4>
                            </div>
                            <div className="overflow-x-auto">
                              <table className="w-full text-sm">
                                <thead className="bg-gray-50">
                                  <tr>
                                    <th className="px-4 py-2 text-left text-gray-600">Date</th>
                                    <th className="px-4 py-2 text-left text-gray-600">Amount</th>
                                    <th className="px-4 py-2 text-left text-gray-600">Type</th>
                                    <th className="px-4 py-2 text-left text-gray-600">Description</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {uploadStatus.extractedData.capitalCalls.slice(0, 5).map((call: any, index: number) => (
                                    <tr key={index} className="border-t border-gray-100">
                                      <td className="px-4 py-2">
                                        {call.call_date ? new Date(call.call_date).toLocaleDateString() : 'N/A'}
                                      </td>
                                      <td className="px-4 py-2 font-medium">
                                        {call.amount ? formatCurrency(call.amount) : 'N/A'}
                                      </td>
                                      <td className="px-4 py-2">{call.call_type || 'N/A'}</td>
                                      <td className="px-4 py-2 text-gray-600 truncate max-w-xs">
                                        {call.description || 'N/A'}
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                            {uploadStatus.extractedData.capitalCalls.length > 5 && (
                              <div className="px-4 py-2 bg-gray-50 text-center text-sm text-gray-600">
                                And {uploadStatus.extractedData.capitalCalls.length - 5} more entries...
                              </div>
                            )}
                          </div>
                        )}

                        {/* Distributions Table */}
                        {uploadStatus.extractedData.distributions && uploadStatus.extractedData.distributions.length > 0 && (
                          <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
                            <div className="bg-gray-50 px-4 py-3 border-b border-gray-200">
                              <h4 className="font-medium text-gray-900 flex items-center">
                                <Table className="w-4 h-4 mr-2" />
                                Distributions ({uploadStatus.extractedData.distributions.length})
                              </h4>
                            </div>
                            <div className="overflow-x-auto">
                              <table className="w-full text-sm">
                                <thead className="bg-gray-50">
                                  <tr>
                                    <th className="px-4 py-2 text-left text-gray-600">Date</th>
                                    <th className="px-4 py-2 text-left text-gray-600">Type</th>
                                    <th className="px-4 py-2 text-left text-gray-600">Amount</th>
                                    <th className="px-4 py-2 text-left text-gray-600">Recallable</th>
                                    <th className="px-4 py-2 text-left text-gray-600">Description</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {uploadStatus.extractedData.distributions.slice(0, 5).map((dist: any, index: number) => (
                                    <tr key={index} className="border-t border-gray-100">
                                      <td className="px-4 py-2">
                                        {dist.distribution_date ? new Date(dist.distribution_date).toLocaleDateString() : 'N/A'}
                                      </td>
                                      <td className="px-4 py-2">{dist.distribution_type || 'N/A'}</td>
                                      <td className="px-4 py-2 font-medium">
                                        {dist.amount ? formatCurrency(dist.amount) : 'N/A'}
                                      </td>
                                      <td className="px-4 py-2">
                                        {dist.is_recallable ? (
                                          <span className="text-green-600">Yes</span>
                                        ) : (
                                          <span className="text-orange-600">No</span>
                                        )}
                                      </td>
                                      <td className="px-4 py-2">{dist.description || 'N/A'}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                            {uploadStatus.extractedData.distributions.length > 5 && (
                              <div className="px-4 py-2 bg-gray-50 text-center text-sm text-gray-600">
                                And {uploadStatus.extractedData.distributions.length - 5} more entries...
                              </div>
                            )}
                          </div>
                        )}

                        {/* Adjustments Table */}
                        {uploadStatus.extractedData.adjustments && uploadStatus.extractedData.adjustments.length > 0 && (
                          <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
                            <div className="bg-gray-50 px-4 py-3 border-b border-gray-200">
                              <h4 className="font-medium text-gray-900 flex items-center">
                                <Table className="w-4 h-4 mr-2" />
                                Adjustments ({uploadStatus.extractedData.adjustments.length})
                              </h4>
                            </div>
                            <div className="overflow-x-auto">
                              <table className="w-full text-sm">
                                <thead className="bg-gray-50">
                                  <tr>
                                    <th className="px-4 py-2 text-left text-gray-600">Date</th>
                                    <th className="px-4 py-2 text-left text-gray-600">Type</th>
                                    <th className="px-4 py-2 text-left text-gray-600">Amount</th>
                                    <th className="px-4 py-2 text-left text-gray-600">Description</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {uploadStatus.extractedData.adjustments.slice(0, 5).map((adj: any, index: number) => (
                                    <tr key={index} className="border-t border-gray-100">
                                      <td className="px-4 py-2">
                                        {adj.adjustment_date ? new Date(adj.adjustment_date).toLocaleDateString() : 'N/A'}
                                      </td>
                                      <td className="px-4 py-2">{adj.adjustment_type || 'N/A'}</td>
                                      <td className="px-4 py-2 font-medium">
                                        {adj.amount ? formatCurrency(adj.amount) : 'N/A'}
                                      </td>
                                      <td className="px-4 py-2">{adj.description || 'N/A'}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                            {uploadStatus.extractedData.adjustments.length > 5 && (
                              <div className="px-4 py-2 bg-gray-50 text-center text-sm text-gray-600">
                                And {uploadStatus.extractedData.adjustments.length - 5} more entries...
                              </div>
                            )}
                          </div>
                        )}

                        {/* No Data Message */}
                        {(!uploadStatus.extractedData.capitalCalls || uploadStatus.extractedData.capitalCalls.length === 0) &&
                         (!uploadStatus.extractedData.distributions || uploadStatus.extractedData.distributions.length === 0) &&
                         (!uploadStatus.extractedData.adjustments || uploadStatus.extractedData.adjustments.length === 0) && (
                          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-center">
                            <p className="text-yellow-800">
                              No structured data was extracted from this document. The document may contain text-only content or the tables may not match expected formats.
                            </p>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="flex gap-3">
                      <a
                        href="/chat"
                        className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition text-sm"
                      >
                        Start Asking Questions
                      </a>
                      <a
                        href="/funds"
                        className="px-4 py-2 bg-white border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition text-sm"
                      >
                        View Fund Dashboard
                      </a>
                      <button
                        onClick={() => {
                          setUploadStatus({ status: "idle" });
                          setUploading(false);
                          clearPersistedData();
                        }}
                        className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition text-sm"
                      >
                        Clear Results
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="mt-12 bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">
          What happens after upload?
        </h2>
        <div className="space-y-4">
          <Step
            number="1"
            title="Document Parsing"
            description="The system uses Docling to extract document structure, identifying tables and text sections."
          />
          <Step
            number="2"
            title="Data Extraction"
            description="Tables containing capital calls, distributions, and adjustments are parsed and stored in the database."
          />
          <Step
            number="3"
            title="Vector Embedding"
            description="Text content is chunked and embedded for semantic search, enabling natural language queries."
          />
          <Step
            number="4"
            title="Ready to Query"
            description="Once processing is complete, you can ask questions about the fund using the chat interface."
          />
        </div>
      </div>
    </div>
  );
}

function Step({
  number,
  title,
  description,
}: {
  number: string;
  title: string;
  description: string;
}) {
  return (
    <div className="flex items-start">
      <div className="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center font-semibold text-sm">
        {number}
      </div>
      <div className="ml-4">
        <h3 className="font-medium text-gray-900">{title}</h3>
        <p className="text-sm text-gray-600 mt-1">{description}</p>
      </div>
    </div>
  );
}
