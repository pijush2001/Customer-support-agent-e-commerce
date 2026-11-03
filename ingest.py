"""Build a local semantic search index from text-based policy PDFs.

Run: python ingest.py --sample
The first embedding call downloads the ONNX MiniLM model. Later calls run locally.
"""
import argparse
import hashlib
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "data" / "policies"
DB_DIR = BASE_DIR / "data" / "chroma"
COLLECTION_NAME = "support_policies_minilm_v1"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MAX_COSINE_DISTANCE = 0.55

SAMPLE_POLICIES = {
    "Return policy": [
        "Returns are accepted within 30 days of delivery for unused items in their original packaging. "
        "Include proof of purchase and all included accessories. This policy applies to purchases from Fictional Demo Store.",
        "To send an item back, contact support with your order number and request a return authorization. "
        "Support supplies the return instructions and a shipping label. Do not send a parcel before authorization. "
        "For a change-of-mind return, the customer pays return shipping. Defective items have free return shipping.",
        "Personalized items, opened hygiene products, and gift cards cannot be returned unless defective. "
        "Items must be inspected before a return is approved. Requesting a return does not automatically approve a refund.",
    ],
    "Refund policy": [
        "Refunds are issued to the original payment method within 5 business days after the returned item is inspected "
        "and the refund is approved. Your bank may take another 3 to 10 business days to display the credit.",
        "Support must verify the order, payment, and return condition before approving a refund. "
        "An AI assistant may explain this policy but cannot approve or issue refunds. A human specialist must approve every refund.",
        "Original shipping charges are not refunded for change-of-mind returns. For an incorrect or defective item, "
        "a specialist may approve shipping reimbursement. Do not promise a specific refund amount before approval.",
    ],
    "Cancellation policy": [
        "You may request cancellation before your order is dispatched. Contact support with your order number as soon "
        "as possible. A cancellation request does not guarantee cancellation because fulfillment may already have begun.",
        "Once an order has been dispatched, it cannot be cancelled. Use the return process after delivery instead. "
        "Personalized orders may not be cancellable after production starts, even if they have not shipped.",
        "Every cancellation requires a human specialist to confirm the fulfillment status and approve the action. "
        "The assistant must not claim an order has been cancelled. Any associated refund follows the refund policy.",
    ],
    "Shipping policy": [
        "Standard shipping takes 3 to 5 business days. Express shipping takes 1 to 2 business days and costs 12 dollars. "
        "Business days exclude weekends and public holidays. Delivery estimates begin after dispatch, not at checkout.",
        "Tracking becomes available after the carrier scans the parcel. For the current status of a specific order, "
        "use the order service rather than this policy document. This document contains no customer order records.",
        "If tracking has not changed for 5 business days, contact support for a carrier investigation. "
        "Weather and carrier disruption can delay delivery. The assistant must not invent a delivery date.",
    ],
    "Damaged item policy": [
        "If an item arrives damaged, we replace it free of charge after verification. Report the damage within "
        "7 days of delivery with your order number and photographs of the item and packaging.",
        "A support ticket routes the report to the fulfillment team. Keep the packaging until the team responds. "
        "Opening a ticket is not confirmation that a replacement has been approved or dispatched.",
        "If a replacement is unavailable, a human specialist will discuss alternatives, including a refund subject "
        "to approval. Our team will follow up by email within 1 business day after a support ticket is opened.",
    ],
    "Account security policy": [
        "To reset your password, open Settings, choose Security, then Reset Password. A reset link is emailed to you "
        "and expires in 30 minutes. If it expires, request a new reset link.",
        "Never share your password, one-time verification code, or full payment card details with support. "
        "Support will not ask you to paste a password into chat. Account changes require identity verification.",
        "If you no longer have access to your registered email or suspect unauthorized account access, contact "
        "a human account-security specialist. The assistant cannot bypass identity verification.",
    ],
}


def generate_sample_pdf(source_dir: Path = SOURCE_DIR) -> Path:
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / "fictional_store_policies.pdf"
    if path.exists():
        return path
    styles = getSampleStyleSheet()
    story = []
    for page_number, (title, paragraphs) in enumerate(SAMPLE_POLICIES.items()):
        if page_number:
            story.append(PageBreak())
        story.append(Paragraph(title, styles["Title"]))
        story.append(Paragraph("FICTIONAL SAMPLE - for development only, not real business policy.", styles["Normal"]))
        story.append(Spacer(1, 16))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, styles["Normal"]))
            story.append(Spacer(1, 12))
    SimpleDocTemplate(str(path)).build(story)
    return path


def load_pdf_chunks(source_dir: Path) -> list[dict]:
    paths = sorted(source_dir.rglob("*.pdf"))
    if not paths:
        raise ValueError(f"No PDFs found in {source_dir}. Run python ingest.py --sample first.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", ".\n", ". ", "\n", " ", ""], keep_separator="end")
    chunks = []
    for path in paths:
        relative_source = path.relative_to(source_dir).as_posix()
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError(f"Encrypted PDF is not supported: {relative_source}")
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                raise ValueError(f"No text on {relative_source} page {page_number}; scanned PDFs require OCR.")
            for chunk_number, content in enumerate(splitter.split_text(text), start=1):
                source = f"{relative_source}:page-{page_number}:chunk-{chunk_number}"
                chunk_id = hashlib.sha256(f"{source}:{file_hash}:{content}".encode()).hexdigest()
                chunks.append({"id": chunk_id, "text": content,
                               "metadata": {"source": relative_source, "page": page_number,
                                            "chunk": chunk_number, "file_sha256": file_hash}})
    return chunks


def open_collection(db_dir: Path = DB_DIR, create: bool = False):
    client = chromadb.PersistentClient(path=str(db_dir))
    embedding_function = ONNXMiniLM_L6_V2()
    if create:
        return client.get_or_create_collection(
            COLLECTION_NAME, embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"})
    try:
        return client.get_collection(COLLECTION_NAME, embedding_function=embedding_function)
    except chromadb.errors.NotFoundError as error:
        raise RuntimeError("Policy index is missing. Run python ingest.py --sample first.") from error


def ingest_documents(source_dir: Path = SOURCE_DIR, db_dir: Path = DB_DIR) -> int:
    chunks = load_pdf_chunks(source_dir)
    collection = open_collection(db_dir, create=True)
    previous_ids = set(collection.get()["ids"])
    current_ids = {chunk["id"] for chunk in chunks}
    for offset in range(0, len(chunks), 64):
        batch = chunks[offset:offset + 64]
        collection.upsert(ids=[chunk["id"] for chunk in batch],
                          documents=[chunk["text"] for chunk in batch],
                          metadatas=[chunk["metadata"] for chunk in batch])
    stale_ids = previous_ids - current_ids
    if stale_ids:
        collection.delete(ids=sorted(stale_ids))
    return collection.count()


def search_policies(question: str, k: int = 2, db_dir: Path = DB_DIR,
                    max_distance: float = MAX_COSINE_DISTANCE) -> list[tuple[str, str]]:
    if not question.strip() or k <= 0:
        return []
    collection = open_collection(db_dir)
    count = collection.count()
    if not count:
        raise RuntimeError("Policy index is empty. Run python ingest.py --sample first.")
    result = collection.query(query_texts=[question], n_results=min(k, count),
                              include=["documents", "metadatas", "distances"])
    matches = []
    for content, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        if distance <= max_distance:
            source = f"{metadata['source']}:page-{metadata['page']}:chunk-{metadata['chunk']}"
            matches.append((source, content))
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="Create the fictional sample PDF if absent")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--db-dir", type=Path, default=DB_DIR)
    args = parser.parse_args()
    if args.sample:
        print(f"Sample PDF: {generate_sample_pdf(args.source_dir)}")
    print(f"Reading PDFs from {args.source_dir}")
    print(f"Chunking: {CHUNK_SIZE} characters, overlap {CHUNK_OVERLAP}; embedding: ONNX MiniLM-L6-v2 (384 dimensions)")
    count = ingest_documents(args.source_dir, args.db_dir)
    print(f"Indexed {count} chunks in {args.db_dir} ({COLLECTION_NAME}, cosine HNSW)")


if __name__ == "__main__":
    main()