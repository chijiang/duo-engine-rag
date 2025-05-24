from asyncore import dispatcher
from typing import Any, List, Optional
from llama_index.core.indices.property_graph.sub_retrievers.llm_synonym import (
    LLMSynonymRetriever,
)
from llama_index.core.async_utils import run_jobs
from llama_index.core.schema import NodeWithScore, QueryBundle, QueryType
from llama_index.core.graph_stores.types import KG_SOURCE_REL
from llama_index.core.indices.property_graph.base import PropertyGraphIndex
from llama_index.core.indices.property_graph.sub_retrievers.base import BasePGRetriever
from llama_index.core.indices.property_graph.retriever import PGRetriever
from llama_index.core.callbacks.schema import CBEventType, EventPayload
import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.events.retrieval import (
    RetrievalEndEvent,
    RetrievalStartEvent,
)
from llama_index.core.graph_stores.types import (
    LabelledNode,
    EntityNode,
    ChunkNode,
)
from llama_index.graph_stores.nebula.utils import remove_empty_values
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore

dispatcher = instrument.get_dispatcher(__name__)


class LLMSynonymRetrieverImpl(LLMSynonymRetriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def aretrieve_from_graph(
        self,
        query_bundle: QueryBundle,
        limit: Optional[int] = None,
        properties: dict = {},
    ) -> List[NodeWithScore]:
        response = await self._llm.apredict(
            self._synonym_prompt,
            query_str=query_bundle.query_str,
            max_keywords=self._max_keywords,
        )
        matches = self._parse_llm_output(response)

        return await self._aprepare_matches(
            matches, limit=limit or self._limit, properties=properties
        )

    async def _aretrieve(
        self, query_bundle: QueryBundle, properties: dict = {}
    ) -> List[NodeWithScore]:
        nodes = await self.aretrieve_from_graph(query_bundle, properties=properties)
        if self.include_text and nodes:
            nodes = await self.async_add_source_text(nodes)
        return nodes

    async def _aprepare_matches(
        self, matches: List[str], limit: Optional[int] = None, properties: dict = {}
    ) -> List[NodeWithScore]:
        kg_nodes = await self._graph_store.aget(ids=matches, properties=properties)
        triplets = await self._graph_store.aget_rel_map(
            kg_nodes,
            depth=self._path_depth,
            limit=limit or self._limit,
            ignore_rels=[KG_SOURCE_REL],
        )

        return self._get_nodes_with_score(triplets)

    @dispatcher.span
    async def aretrieve(
        self, str_or_query_bundle: QueryType, properties: dict = {}
    ) -> List[NodeWithScore]:
        self._check_callback_manager()

        dispatcher.event(
            RetrievalStartEvent(
                str_or_query_bundle=str_or_query_bundle,
            )
        )
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = await self._aretrieve(
                    query_bundle=query_bundle, properties=properties
                )
                nodes = await self._ahandle_recursive_retrieval(
                    query_bundle=query_bundle, nodes=nodes
                )
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        dispatcher.event(
            RetrievalEndEvent(
                str_or_query_bundle=str_or_query_bundle,
                nodes=nodes,
            )
        )
        return nodes


class PGRetrieverImpl(PGRetriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def _aretrieve(
        self, query_bundle: QueryBundle, properties: dict = {}
    ) -> List[NodeWithScore]:
        tasks = []
        for sub_retriever in self.sub_retrievers:
            if isinstance(sub_retriever, LLMSynonymRetrieverImpl):
                tasks.append(
                    sub_retriever.aretrieve(query_bundle, properties=properties)
                )
            else:
                tasks.append(sub_retriever.aretrieve(query_bundle))

        async_results = await run_jobs(
            tasks, workers=self.num_workers, show_progress=self.show_progress
        )

        # flatten the results
        return self._deduplicate([node for nodes in async_results for node in nodes])

    @dispatcher.span
    async def aretrieve(
        self, str_or_query_bundle: QueryType, properties: dict = {}
    ) -> List[NodeWithScore]:
        self._check_callback_manager()

        dispatcher.event(
            RetrievalStartEvent(
                str_or_query_bundle=str_or_query_bundle,
            )
        )
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = await self._aretrieve(
                    query_bundle=query_bundle, properties=properties
                )
                nodes = await self._ahandle_recursive_retrieval(
                    query_bundle=query_bundle, nodes=nodes
                )
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        dispatcher.event(
            RetrievalEndEvent(
                str_or_query_bundle=str_or_query_bundle,
                nodes=nodes,
            )
        )
        return nodes


class PropertyGraphIndexImpl(PropertyGraphIndex):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def as_retriever(
        self,
        sub_retrievers: Optional[List["BasePGRetriever"]] = None,
        include_text: bool = True,
        **kwargs: Any,
    ) -> PGRetrieverImpl:

        from llama_index.core.indices.property_graph.sub_retrievers.vector import (
            VectorContextRetriever,
        )

        if sub_retrievers is None:
            sub_retrievers = [
                LLMSynonymRetrieverImpl(
                    graph_store=self.property_graph_store,
                    include_text=include_text,
                    llm=self._llm,
                    **kwargs,
                ),
            ]

            if self._embed_model and (
                self.property_graph_store.supports_vector_queries or self.vector_store
            ):
                sub_retrievers.append(
                    VectorContextRetriever(
                        graph_store=self.property_graph_store,
                        vector_store=self.vector_store,
                        include_text=include_text,
                        embed_model=self._embed_model,
                        **kwargs,
                    )
                )

        return PGRetrieverImpl(sub_retrievers, use_async=self._use_async, **kwargs)


class NebulaPropertyGraphStoreImpl(NebulaPropertyGraphStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes."""
        cypher_statement = "MATCH (e:Node__) "
        if properties or ids:
            cypher_statement += "WHERE "
        params = {}

        if ids:
            cypher_statement += f"id(e) in $all_id "
            params[f"all_id"] = ids
        if properties:
            properties_list = []
            for i, prop in enumerate(properties):
                properties_list.append(f"e.Props__.`{prop}` == $property_{i}")
                params[f"property_{i}"] = properties[prop]
            cypher_statement += " AND " + " AND ".join(properties_list)

        return_statement = """
        RETURN id(e) AS name,
               e.Node__.label AS type,
               properties(e.Props__) AS properties,
               properties(e) AS all_props
        """
        cypher_statement += return_statement
        cypher_statement = cypher_statement.replace("\n", " ")

        response = self.structured_query(cypher_statement, param_map=params)

        nodes = []
        for record in response:
            if "text" in record["all_props"]:
                node = ChunkNode(
                    id_=record["name"],
                    label=record["type"],
                    text=record["all_props"]["text"],
                    properties=remove_empty_values(record["properties"]),
                )
            elif "name" in record["all_props"]:
                node = EntityNode(
                    id_=record["name"],
                    label=record["type"],
                    name=record["all_props"]["name"],
                    properties=remove_empty_values(record["properties"]),
                )
            else:
                node = EntityNode(
                    name=record["name"],
                    type=record["type"],
                    properties=remove_empty_values(record["properties"]),
                )
            nodes.append(node)
        return nodes
