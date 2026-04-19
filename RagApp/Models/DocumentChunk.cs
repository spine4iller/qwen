namespace RagApp.Models;

public class DocumentChunk
{
    public string Content { get; set; } = string.Empty;
    public string Source { get; set; } = string.Empty;
    public float[] Embedding { get; set; } = Array.Empty<float>();
}

public class QueryResult
{
    public string Answer { get; set; } = string.Empty;
    public List<DocumentChunk> RelevantChunks { get; set; } = new();
}
