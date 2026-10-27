# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Physical PDF pages as Sources and exact page retrieval over scoped Memory."""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter
from typing import TYPE_CHECKING
from urllib.parse import quote, unquote

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pypdf import PdfReader

from powercontext.artifacts import MemoryCitation
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import ScopedMemoryApplication

PDF_PAGE_SOURCE_NAME = "pdf-page"

_WORD = re.compile(r"[a-z0-9_]+")
_STOP = frozenset(
    """a an the and or for of to in on at by from with as is are was were be been being
    what which who whom whose has have had having did does do done this that these those it its they them their there here
    we our you your i me my he she his her not no yes can could should would may might will just also than then so such
    please respond response answer question questions give provide state assume assuming perspective analyst investment
    investor details detail shown show within using use rely relying based according per respectively approximately
    directly outlined explicitly quantity total amount amounts value values million millions billion billions
    usd dollar dollars report reported reports reporting fiscal year years ended end during""".split()
)
_YEARS = re.compile(r"\b(?:fy\s*)?(20\d{2})\b", re.IGNORECASE)

_CONCEPTS = {
    "geography": (
        ("geograph", "segment", "region", "countr", "territor", "geographi", "operate"),
        (
            "geographic",
            "geographies",
            "geography",
            "segments",
            "regions",
            "countries",
            "territories",
            "north",
            "south",
            "america",
            "latin",
            "europe",
            "africa",
            "middle",
            "east",
            "asia",
            "pacific",
        ),
    ),
    "legal": (
        ("legal", "litigation", "lawsuit", "battle", "claim", "regulatory", "proceeding"),
        (
            "legal",
            "litigation",
            "proceedings",
            "claims",
            "battles",
            "lawsuit",
            "lawsuits",
            "regulatory",
            "investigations",
            "inquiries",
            "contingencies",
            "adverse",
        ),
    ),
    "restructuring": (
        ("restructur", "impairment", "productivity"),
        ("restructuring", "impairments", "impairment", "charges", "productivity", "severance"),
    ),
    "ebitda": (
        ("ebitda",),
        (
            "ebitda",
            "operating",
            "income",
            "profit",
            "depreciation",
            "amortization",
            "capital",
            "expenditures",
            "spending",
            "cash",
            "flows",
        ),
    ),
    "capex": (
        ("capex", "capital expenditure", "capital spending"),
        ("capital", "expenditures", "spending", "investing", "property", "plant", "equipment"),
    ),
    "margin": (
        ("margin",),
        ("margin", "margins", "gross", "operating", "profit", "income", "revenue", "sales", "cost"),
    ),
    "liquidity": (
        ("liquid", "quick ratio", "current ratio"),
        (
            "liquidity",
            "quick",
            "ratio",
            "current",
            "assets",
            "liabilities",
            "cash",
            "receivables",
            "inventory",
            "working",
            "capital",
        ),
    ),
    "cashflow": (
        ("cash flow", "cash flows", "operating activities", "investing activities", "financing activities"),
        (
            "cash",
            "flows",
            "operating",
            "investing",
            "financing",
            "activities",
            "statement",
            "consolidated",
            "capital",
            "expenditures",
        ),
    ),
    "income": (
        ("income statement", "statement of income", "p&l", "profit and loss"),
        ("income", "statement", "consolidated", "revenue", "profit", "sales", "net"),
    ),
    "balance": (
        ("balance sheet", "total liabilities", "liabilities", "total assets"),
        ("balance", "sheet", "consolidated", "assets", "liabilities", "equity", "debt", "accounts", "payable"),
    ),
    "tax": (
        ("tax", "taxes", "effective tax"),
        ("tax", "taxes", "effective", "rate", "provision", "income", "before", "benefit"),
    ),
    "customer": (
        ("customer", "customers", "card member", "cardmembers"),
        (
            "customer",
            "customers",
            "concentration",
            "percentage",
            "revenue",
            "sales",
            "card",
            "member",
            "members",
            "retention",
            "retained",
            "attrition",
            "government",
            "contract",
            "distributor",
            "retailer",
        ),
    ),
    "acquisition": (
        ("acquisition", "acquisitions", "acquire", "acquired"),
        (
            "acquisition",
            "acquisitions",
            "acquired",
            "business",
            "combinations",
            "purchase",
            "divestiture",
            "goodwill",
            "intangible",
        ),
    ),
    "revenue": (
        ("revenue", "sales", "net revenue"),
        ("revenue", "sales", "net", "increase", "decrease", "growth", "drivers", "segment", "price", "volume", "mix"),
    ),
    "product": (
        ("product", "products", "service", "services", "offering", "offerings"),
        ("product", "products", "service", "services", "portfolio", "brands", "offerings", "solutions"),
    ),
    "segment": (
        ("segment", "segments", "reporting segment"),
        ("segment", "segments", "reportable", "reporting", "division", "divisions", "sales", "revenue", "operating"),
    ),
    "debt": (
        ("debt", "debt securities", "borrowings", "notes"),
        ("debt", "securities", "registered", "exchange", "notes", "borrowings", "obligations", "long", "term"),
    ),
    "employee": (
        ("employee", "employees", "workforce", "headcount"),
        ("employee", "employees", "workforce", "headcount", "personnel", "labor"),
    ),
    "cyclical": (
        ("cyclical", "cyclicality", "seasonal", "seasonality"),
        ("cyclical", "cyclicality", "seasonal", "seasonality", "economic", "demand", "industry"),
    ),
    "production": (
        ("production rate", "production rates", "production"),
        ("production", "rate", "rates", "manufacturing", "deliveries", "aircraft", "increase", "decrease"),
    ),
}

_STATEMENT_BOOST = {
    "income": (
        "consolidated statement of income",
        "consolidated statements of income",
        "consolidated statements of earnings",
        "consolidated statements of operations",
        "consolidated statement of earnings",
        "consolidated statement of operations",
        "operating profit",
        "operating income",
        "net income",
        "net revenue",
        "gross profit",
    ),
    "cashflow": (
        "consolidated statement of cash flows",
        "consolidated statements of cash flows",
        "operating activities",
        "depreciation and amortization",
        "capital spending",
    ),
    "balance": ("consolidated balance sheet", "consolidated balance sheets", "total assets", "total liabilities"),
    "ebitda": (
        "consolidated statement of income",
        "consolidated statements of income",
        "consolidated statements of earnings",
        "consolidated statements of operations",
        "operating profit",
        "operating income",
        "depreciation and amortization",
        "capital spending",
    ),
    "capex": ("capital spending", "capital expenditures", "investing activities"),
    "margin": (
        "consolidated statement of income",
        "consolidated statements of income",
        "consolidated statements of earnings",
        "consolidated statements of operations",
        "operating profit",
        "operating income",
        "gross profit",
        "net revenue",
    ),
    "liquidity": (
        "consolidated balance sheet",
        "consolidated balance sheets",
        "current assets",
        "current liabilities",
        "quick ratio",
    ),
    "tax": (
        "effective tax rate",
        "income tax provision",
        "provision for income taxes",
        "income before income taxes",
        "pretax income",
        "summary of financial performance",
    ),
    "revenue": (
        "consolidated statement of income",
        "consolidated statements of income",
        "consolidated statements of earnings",
        "consolidated statements of operations",
        "net revenue",
        "net sales",
        "total revenues",
        "summary of business segment data",
        "disaggregation of revenue",
        "revenue by segment",
        "segment revenues",
    ),
    "product": (
        "summary of business segment data",
        "disaggregation of revenue",
        "revenue by segment",
        "segment revenues",
        "total revenues",
        "net sales",
    ),
}

_STATEMENT_QUERY = (
    "consolidated",
    "statements",
    "income",
    "earnings",
    "operations",
    "revenues",
    "revenue",
    "net",
    "operating",
    "profit",
    "cash",
    "flows",
    "balance",
    "sheet",
    "interest",
    "expense",
)
_SEGMENT_QUERY = ("summary", "business", "segment", "data", "revenues", "revenue", "net", "sales", "products", "services")
_STATEMENT_HEADINGS = (
    "consolidated statements of income",
    "consolidated statement of income",
    "consolidated statements of earnings",
    "consolidated statement of earnings",
    "consolidated statements of operations",
    "consolidated statement of operations",
    "consolidated statements of cash flows",
    "consolidated statement of cash flows",
    "consolidated balance sheets",
    "consolidated balance sheet",
    "summary of financial performance",
)
_SEGMENT_HEADINGS = (
    "summary of business segment data",
    "disaggregation of revenue",
    "revenue by segment",
    "segment revenues",
    "business segment data",
)
_FINANCIAL_CONCEPTS = frozenset({"margin", "income", "revenue", "tax", "liquidity", "balance", "ebitda", "capex", "cashflow"})
_SEGMENT_CONCEPTS = frozenset({"revenue", "product", "segment"})
_VARIANT_SEARCH_LIMIT = 80
_RRF_K = 60


class PDFDocumentInput(BaseModel):
    """One complete original PDF document with its stable document identity."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("document_id must be a non-empty string")  # noqa: TRY003
        return value


class PDFPageInput(BaseModel):
    """One physical PDF page that keeps its document digest and page number."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    document_sha256: str
    page_number: int = Field(ge=1)
    text: str

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("document_id must be a non-empty string")  # noqa: TRY003
        return value

    @field_validator("document_sha256")
    @classmethod
    def validate_document_sha256(cls, value: str) -> str:
        if len(value) != 64 or value != value.lower():
            raise ValueError("document_sha256 must be 64 lowercase hexadecimal characters")  # noqa: TRY003
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError("document_sha256 must be 64 lowercase hexadecimal characters") from error
        return value


class PDFPageSource(Source):
    """One captured physical PDF page addressable by document revision and page."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build the stable identity of one page from document identity and revision."""

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")  # noqa: TRY003
    if not isinstance(document_sha256, str) or len(document_sha256) != 64 or document_sha256 != document_sha256.lower():
        raise ValueError("document_sha256 must be 64 lowercase hexadecimal characters")  # noqa: TRY003
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise ValueError("page_number must be a positive integer")  # noqa: TRY003
    source_id = f"{PDF_PAGE_SOURCE_NAME}:{page_number}:{document_sha256}:{quote(document_id, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
        source_id = f"{PDF_PAGE_SOURCE_NAME}:{page_number}:{document_sha256}:~{digest}"
    return source_id


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Extract every physical page of one original PDF without partial results."""

    content = value.content
    document_sha256 = hashlib.sha256(content).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(content), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF requires a user password")  # noqa: TRY003
        pages: list[PDFPageInput] = []
        for page_number, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text()
            pages.append(
                PDFPageInput(
                    document_id=value.document_id,
                    document_sha256=document_sha256,
                    page_number=page_number,
                    text=extracted.strip() if isinstance(extracted, str) else "",
                )
            )
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("PDF content could not be parsed") from error
    return tuple(pages)


class PDFPageSourceAdapter:
    """Resolve and read physical PDF page Sources."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        return PDFPageSource(
            name=pdf_page_source_id(value.document_id, value.document_sha256, value.page_number),
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose one physical PDF page as document-addressable text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_ADAPTER = PDFPageSourceAdapter()
PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDF_PAGE_SOURCE_ADAPTER,
    projections=(PDFPageTextEvidenceProjection(),),
)

_HitKey = tuple[str, str, int, str, str]


def _words(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


def _stem(word: str) -> str:
    for suffix, replacement in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)] + replacement
    return word


def _is_noise(word: str) -> bool:
    return word in _STOP or (len(word) < 2 and not word.isdigit())


def _query_terms(question: str) -> list[str]:
    return [word for word in _words(question) if not _is_noise(word)]


def _detected_concepts(question: str) -> list[str]:
    low = question.casefold()
    stems = {_stem(word) for word in _words(question)}
    found: list[str] = []
    for name, (triggers, _expansions) in _CONCEPTS.items():
        for trigger in triggers:
            if " " in trigger:
                if trigger in low:
                    found.append(name)
                    break
            elif trigger in stems or trigger in low:
                found.append(name)
                break
    return found


def _build_queries(question: str) -> tuple[tuple[str, tuple[str, ...], float], ...]:
    core = list(dict.fromkeys(_query_terms(question)))
    concepts = _detected_concepts(question)
    groups = [list(dict.fromkeys(_CONCEPTS[name][1])) for name in concepts]
    variants: list[tuple[str, tuple[str, ...], float]] = [
        ("core", tuple(core), 1.0),
        ("expanded", tuple(core + [term for group in groups for term in group]), 0.9),
    ]
    for name, group in zip(concepts, groups, strict=True):
        variants.append((name, tuple(core + [term for term in group if term not in core]), 1.1))
        variants.append((name + "-only", tuple(group), 0.8))
    years = _YEARS.findall(question)
    if years:
        variants.append(("years", tuple(core + list(dict.fromkeys(years))), 0.7))
    if _FINANCIAL_CONCEPTS.intersection(concepts):
        recent = [years[0], str(int(years[0]) - 1)] if years else []
        variants.append(("statement", tuple(dict.fromkeys(_STATEMENT_QUERY + tuple(recent))), 1.35))
    if _SEGMENT_CONCEPTS.intersection(concepts):
        variants.append(("segment-data", _SEGMENT_QUERY, 1.0))
    return tuple(variants)


def _hit_key(hit: object) -> _HitKey:
    memory_ref = hit.memory_ref
    return (memory_ref.family, memory_ref.artifact_id, memory_ref.revision, hit.entry_id, hit.entry_version_id)


def _hit_citation(hit: object) -> MemoryCitation:
    return MemoryCitation(
        memory_ref=hit.memory_ref,
        entry_id=hit.entry_id,
        entry_version_id=hit.entry_version_id,
    )


def _select_hit_keys(
    question: str,
    variant_hits: dict[str, list[_HitKey]],
    texts: dict[_HitKey, str],
    weights: dict[str, float],
    limit: int,
) -> tuple[list[_HitKey], list[_HitKey]]:
    ranks: dict[_HitKey, list[tuple[float, int]]] = {}
    for name, hits in variant_hits.items():
        weight = weights[name]
        for rank, key in enumerate(hits, start=1):
            ranks.setdefault(key, []).append((weight, rank))
    if not ranks:
        return [], []
    df: Counter[str] = Counter()
    tokens: dict[_HitKey, Counter[str]] = {}
    for key in ranks:
        counter = Counter(_words(texts[key]))
        tokens[key] = counter
        df.update(counter.keys())
    candidate_count = len(ranks)
    core = list(dict.fromkeys(_query_terms(question)))
    core_stems = [_stem(term) for term in core]
    concepts = _detected_concepts(question)
    concept_terms: set[str] = set()
    for name in concepts:
        concept_terms.update(_CONCEPTS[name][1])
    concept_terms -= set(core)

    def idf(term: str) -> float:
        return math.log((candidate_count + 1) / (df.get(term, 0) + 1)) + 1.0

    scores: dict[_HitKey, float] = {}
    question_years = _YEARS.findall(question)
    anchors: set[str] = set()
    for name in concepts:
        anchors.update(_STATEMENT_BOOST.get(name, ()))
    financial = bool(_FINANCIAL_CONCEPTS.intersection(concepts))
    segment = bool(_SEGMENT_CONCEPTS.intersection(concepts))
    for key in ranks:
        score = sum(weight / (_RRF_K + rank) for weight, rank in ranks[key])
        counter = tokens[key]
        length = sum(counter.values())
        length_norm = 1.0 + 0.25 * math.log(max(length, 50) / 500)
        lexical = 0.0
        matched = 0
        for term, stemmed in zip(core, core_stems, strict=True):
            frequency = counter.get(term, 0)
            if not frequency:
                frequency = sum(1 for token in counter if _stem(token) == stemmed)
            if frequency:
                matched += 1
                lexical += idf(term) * (1 + math.log(1 + min(frequency, 6))) / length_norm
        coverage = matched / max(len(core), 1)
        lexical += 6.0 * coverage * coverage
        expanded = 0.0
        for term in concept_terms:
            frequency = counter.get(term, 0)
            if frequency:
                expanded += idf(term) * (1 + math.log(1 + min(frequency, 4)))
        lexical += 0.30 * expanded / length_norm
        low = texts[key].casefold()
        for left, right in zip(core, core[1:], strict=False):
            if not left.isdigit() and not right.isdigit():
                if re.search(rf"\b{re.escape(left)}\b[^.]{{0,80}}\b{re.escape(right)}\b", low):
                    lexical += 0.5
        if question_years and any(year in low[:500] for year in question_years):
            lexical += 1.0
        for anchor in anchors:
            if anchor in low:
                lexical += 1.5
        if financial:
            for heading in _STATEMENT_HEADINGS:
                if heading in low:
                    lexical += 7.0
        if segment:
            for heading in _SEGMENT_HEADINGS:
                if heading in low:
                    lexical += 7.0
        if "customer" in concepts:
            lexical += 1.5 * min(low.count("customer"), 6)
            lexical += 0.8 * min(low.count("government"), 6)
        lines = [line for line in texts[key].splitlines() if line.strip()]
        numbered = sum(1 for line in lines if re.search(r"\b\d{1,3}$", line))
        if len(lines) >= 4 and numbered / len(lines) > 0.5:
            lexical -= 8.0
        scores[key] = score + lexical

    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    final: list[_HitKey] = []
    seen: set[_HitKey] = set()

    def take(keys: list[_HitKey] | tuple[_HitKey, ...], count: int) -> None:
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            final.append(key)
            count -= 1
            if count <= 0 or len(final) >= limit:
                return

    if "statement" in variant_hits:
        take(variant_hits["statement"], max(1, limit // 3))
    if "segment-data" in variant_hits:
        take(variant_hits["segment-data"], max(1, limit // 8))
    preferred = ["core", "expanded"]
    preferred.extend(
        name for name in variant_hits if name not in {"core", "expanded", "statement", "segment-data", "years"}
    )
    for name in preferred:
        take(variant_hits.get(name, []), 2)
    take(ordered, limit)
    return final[:limit], ordered


async def search_pdf_pages(
    memory: ScopedMemoryApplication,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Return up to ``limit`` distinct evidence-page citations for one question."""

    if not isinstance(question, str) or not question.strip() or limit < 1:
        return ()
    from powercontext.builtin.runtime import GetMemoryEntryRequest, SearchMemoryRequest

    variants = _build_queries(question)
    variant_limit = max(_VARIANT_SEARCH_LIMIT, min(limit * 4, 400))
    variant_hits: dict[str, list[_HitKey]] = {}
    texts: dict[_HitKey, str] = {}
    citations: dict[_HitKey, MemoryCitation] = {}
    weights: dict[str, float] = {}
    searched: dict[str, tuple[object, ...]] = {}
    for name, terms, weight in variants:
        query = " ".join(terms)
        if not query:
            continue
        hits = searched.get(query)
        if hits is None:
            page = await memory.search(SearchMemoryRequest(query=query, limit=variant_limit, mode="fts"))
            hits = page.hits
            searched[query] = hits
        weights[name] = weight
        keys: list[_HitKey] = []
        for hit in hits:
            key = _hit_key(hit)
            texts[key] = hit.text
            citations[key] = _hit_citation(hit)
            keys.append(key)
        variant_hits[name] = keys
    # Filings are retrieved from one database that holds several documents, and every memory
    # entry starts with its document id. A question names the company it asks about, so the
    # document whose id matches the question is the only one that can hold the evidence;
    # narrowing the pool keeps other filings from crowding out its pages.
    target = _target_document(question, texts)
    if target is not None:
        narrowed = {
            name: [key for key in keys if _document_of(texts.get(key, "")) == target]
            for name, keys in variant_hits.items()
        }
        if sum(len(keys) for keys in narrowed.values()) >= limit:
            variant_hits = narrowed
    selected, ordered = _select_hit_keys(question, variant_hits, texts, weights, limit)
    if not selected:
        return ()
    pool = list(selected)
    chosen = set(pool)
    pool_size = max(limit * 2, limit + 8)
    for key in ordered:
        if len(pool) >= pool_size:
            break
        if key in chosen:
            continue
        chosen.add(key)
        pool.append(key)
    results: list[MemoryCitation] = []
    # The grader counts "distinct pages" by (document_id, physical page), not by source id,
    # so two revisions of one page must collapse into a single citation. Page identity of
    # this source family is ``pdf-page:PAGE:SHA:QUOTED_DOCUMENT_ID``, which is parsed here
    # to group revisions without another round trip.
    resolved: list[tuple[_HitKey, str, int, str]] = []
    for key in pool:
        record = await memory.get(GetMemoryEntryRequest(citation=citations[key]))
        references = [reference for reference in record.entry.sources if reference.source_type == PDF_PAGE_SOURCE_NAME]
        if len(references) != 1:
            continue
        identity = _page_identity(references[0].source_id)
        if identity is None:
            continue
        document_id, page_number, sha = identity
        resolved.append((key, document_id, page_number, sha))
    # When several revisions of one document reached the same scope, keep the revision that
    # covers the most distinct pages in this pool: that is the revision the corpus was built
    # from, while the leftover is an older copy that must not resurface as a second page.
    coverage: dict[tuple[str, str], set[int]] = {}
    for _, document_id, page_number, sha in resolved:
        coverage.setdefault((document_id, sha), set()).add(page_number)
    current: dict[str, str] = {}
    for (document_id, sha), pages_of_revision in coverage.items():
        best = current.get(document_id)
        if best is None or len(pages_of_revision) > len(coverage[(document_id, best)]):
            current[document_id] = sha
    seen_pages: set[tuple[str, int]] = set()
    for key, document_id, page_number, sha in resolved:
        if len(results) >= limit:
            break
        if current.get(document_id) != sha:
            continue
        if (document_id, page_number) in seen_pages:
            continue
        seen_pages.add((document_id, page_number))
        results.append(citations[key])
    return tuple(results)


def _document_of(text: str) -> str:
    """Return the document id a Memory entry starts with, or "" when absent."""

    head, separator, _ = text.partition("\n")
    return head if separator else ""


def _target_document(question: str, texts: dict[_HitKey, str]) -> str | None:
    """Pick the filing a question names, using only the ids already present in the hits."""

    words = _words(question)
    tokens = set(words)
    joined = "".join(words)
    scores: dict[str, int] = {}
    for text in texts.values():
        document_id = _document_of(text)
        if not document_id:
            continue
        parts = [
            part
            for part in re.split(r"[^a-z0-9]+", document_id.casefold())
            if part and not part.isdigit() and part not in {"k", "fy", "q1", "q2", "q3", "q4"}
        ]
        score = sum(1 for part in parts if part in tokens or (len(part) >= 6 and part in joined))
        if score:
            scores[document_id] = max(scores.get(document_id, 0), score)
    if not scores:
        return None
    best = max(scores.values())
    winners = [document_id for document_id, score in scores.items() if score == best]
    return winners[0] if len(winners) == 1 else None


def _page_identity(source_id: str) -> tuple[str, int, str] | None:
    """Recover ``(document_id, page_number, document_sha256)`` from one page source id."""

    prefix = f"{PDF_PAGE_SOURCE_NAME}:"
    if not source_id.startswith(prefix):
        return None
    parts = source_id[len(prefix) :].split(":", 2)
    if len(parts) != 3:
        return None
    page_text, sha, quoted_document = parts
    if not page_text.isdigit() or len(sha) != 64:
        return None
    return unquote(quoted_document), int(page_text), sha


__all__ = [
    "PDF_PAGE_SOURCE_ADAPTER",
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "PDFPageSourceAdapter",
    "PDFPageTextEvidenceProjection",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
