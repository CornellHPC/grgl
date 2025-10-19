/* Genotype Representation Graph Library (GRGL)
 * Copyright (C) 2024 April Wei
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "grgl/map_mutations.h"
#include "grg_helpers.h"
#include "grgl/common.h"
#include "grgl/csr_storage.h"
#include "grgl/grgnode.h"
#include "grgl/mut_iterator.h"
#include "grgl/mutation.h"
#include "grgl/visitor.h"
#include "util.h"
#include <algorithm>
#include <atomic>
#include <chrono>

#include <cstddef>
#include <iostream>
#include <omp.h>
#include <unordered_map>
#include <vector>

#include <thread>
// When enabled: garbage collects unneeded sample sets
#define CLEANUP_SAMPLE_SETS_MAPPING 1

// Every 10% we'll emit some stats about current size of the GRG.
#define EMIT_STATS_AT_PERCENT (10)
// Every 5% we compact the GRG edge lists.
#define COMPACT_EDGES_AT_PERCENT (5)
#define ONE_HUNDRED_PERCENT      (100)

// Histogram size for statistics
#define STATS_HIST_SIZE (200)

namespace grgl {

using NodeSamples = std::tuple<NodeID, size_t>;

static bool cmpNodeSamples(const NodeSamples& ns1, const NodeSamples& ns2) {
    const size_t samplesCount1 = std::get<1>(ns1);
    const size_t& samplesCount2 = std::get<1>(ns2);
    return samplesCount1 < samplesCount2;
}

class TopoCandidateCollectorVisitor : public grgl::GRGVisitor {
public:
    explicit TopoCandidateCollectorVisitor(const std::vector<NodeIDSizeT>& sampleCounts)
        : m_sampleCounts(sampleCounts) {}

    bool visit(const grgl::GRGPtr& grg,
               const grgl::NodeID nodeId,
               const grgl::TraversalDirection direction,
               const grgl::DfsPass dfsPass = grgl::DfsPass::DFS_PASS_NONE) override {
        release_assert(direction == TraversalDirection::DIRECTION_UP);
        release_assert(dfsPass == DfsPass::DFS_PASS_NONE);
        release_assert(grg->hasUpEdges());

#if CLEANUP_SAMPLE_SETS_MAPPING
        if (m_refCounts.empty()) {
            m_refCounts.resize(grg->numNodes());
        }
#endif
        const bool isRoot = grg->numUpEdges(nodeId) == 0;
        const bool isSample = grg->isSample(nodeId);
        const size_t ploidy = grg->getPloidy();
        const auto numCoals = grg->getNumIndividualCoals(nodeId);
        const bool computeCoals = !isSample && (ploidy == 2) && (COAL_COUNT_NOT_SET == numCoals);

        size_t individualCoalCount = 0;
        // Map from an individual to which child contained it.
        std::unordered_map<NodeIDSizeT, NodeIDSizeT> individualToChild;
        NodeIDList candidateNodes;
        NodeIDList samplesBeneath;
        if (isSample) {
            samplesBeneath.emplace_back(nodeId);
        }
#if CLEANUP_SAMPLE_SETS_MAPPING
        m_refCounts[nodeId] = grg->numUpEdges(nodeId);
#endif
        for (const auto& childId : grg->getDownEdges(nodeId)) {
            const auto& childSampleIt = m_nodeToSamples.find(childId);
            if (childSampleIt != m_nodeToSamples.end()) {
                auto childSamples = childSampleIt->second;
                if (childSamples.size() > 1) {
                    candidateNodes.emplace_back(childId);
                }
                for (const auto childSampleId : childSamples) {
                    samplesBeneath.emplace_back(childSampleId);
                    if (computeCoals) {
                        auto insertPair = individualToChild.emplace(childSampleId / ploidy, childId);
                        // The individual already existed from a _different child_, so the two samples just coalesced.
                        if (!insertPair.second && childId != insertPair.first->second) {
                            individualCoalCount++;
                        }
                    }
                }
            }
        }
        // Check if we had a mismatch in expected vs. total sample sets.
        release_assert(nodeId < m_sampleCounts.size());
        release_assert(m_sampleCounts[nodeId] <= grg->numSamples());
        NodeIDSizeT missing = (m_sampleCounts[nodeId] - samplesBeneath.size());

        // We can only record coalescence counts if there are no samples missing.
        if (missing == 0 && computeCoals) {
            grg->setNumIndividualCoals(nodeId, individualCoalCount);
        }

        // If we've reached the root of the graph or have missing samples beneath us, we need to stop the search
        // and emit candidate nodes to map the mutation to.
        const bool keepGoing = (missing == 0 && !isRoot);
        if (missing == 0 && isRoot) {
            m_collectedNodes.emplace_back(nodeId, samplesBeneath.size()); // Root is a candidate node.
#if CLEANUP_SAMPLE_SETS_MAPPING
            // Prevent candidates from having their samplesets garbage collected.
            m_refCounts[nodeId] = MAX_GRG_NODES + 1;
#endif
            m_nodeToSamples.emplace(nodeId, std::move(samplesBeneath));
        } else if (!keepGoing) {
            for (const auto& candidate : candidateNodes) {
                m_collectedNodes.emplace_back(candidate, m_nodeToSamples[candidate].size());
#if CLEANUP_SAMPLE_SETS_MAPPING
                // Prevent candidates from having their samplesets garbage collected.
                m_refCounts[candidate] = MAX_GRG_NODES + 1;
#endif
            }
        } else {
            m_nodeToSamples.emplace(nodeId, std::move(samplesBeneath));
        }

#if CLEANUP_SAMPLE_SETS_MAPPING
        for (const auto& childId : grg->getDownEdges(nodeId)) {
            // Skip children that aren't part of our search.
            if (m_refCounts[childId] == 0) {
                continue;
            }
            if (--m_refCounts[childId] == 0) {
                m_nodeToSamples.erase(childId);
            }
        }
#endif
        return keepGoing;
    }

    NodeIDList getSamplesForCandidate(NodeID candidateId) {
        NodeIDList result;
        auto findIt = m_nodeToSamples.find(candidateId);
        release_assert(findIt != m_nodeToSamples.end());
        result = std::move(findIt->second);
        m_nodeToSamples.erase(findIt);
        return std::move(result);
    }

    std::vector<NodeSamples> m_collectedNodes;
    std::unordered_map<NodeID, NodeIDList> m_nodeToSamples;

private:
    // These are the _total_ samples beneath each node (not restricted to current samples being searched)
    const std::vector<NodeIDSizeT>& m_sampleCounts;
#if CLEANUP_SAMPLE_SETS_MAPPING
    std::vector<NodeIDSizeT> m_refCounts;
#endif
};

static bool setsOverlap(const NodeIDSet& alreadyCovered, const NodeIDList& candidateSet) {
    for (auto nodeId : candidateSet) {
        if (alreadyCovered.find(nodeId) != alreadyCovered.end()) {
            return true;
        }
    }
    return false;
}

static std::pair<NodeIDList, NodeIDSizeT> greedyAddMutationImmutable(const MutableGRGPtr& grg,
                                                                     const std::vector<NodeIDSizeT>& sampleCounts,
                                                                     const NodeIDList& mutSamples,
                                                                     MutationMappingStats& stats,
                                                                     const size_t batchCount) {
    // The topological order of nodeIDs is maintained through-out this algorithm, because newly added
    // nodes are only ever _root nodes_ (at the time they are added).
    release_assert(grg->nodesAreOrdered());

    const size_t ploidy = grg->getPloidy();
    // The set of nodes that we have covered so far (greedily extended)
    NodeIDSet covered;

    TopoCandidateCollectorVisitor collector(sampleCounts);
    if (mutSamples.size() > 1) {
        grg->visitTopo(collector, grgl::TraversalDirection::DIRECTION_UP, mutSamples);
    }
    std::vector<NodeSamples>& candidates = collector.m_collectedNodes;
    std::sort(candidates.begin(), candidates.end());
    auto endOfUnique = std::unique(candidates.begin(), candidates.end());
    candidates.erase(endOfUnique, candidates.end());
    std::sort(candidates.begin(), candidates.end(), cmpNodeSamples);

    if (candidates.empty()) {
        stats.mutationsWithNoCandidates++;
    } else {
        // Exact match scenario. Return early.
        const auto& candidate = candidates.back();
        const size_t candidateSetSize = std::get<1>(candidate);
        if (candidateSetSize == mutSamples.size()) {
            const auto candidateId = std::get<0>(candidate);
            stats.reusedExactly++;
            // Note that, because of our batching scheme, reusedMutNodes is exactly zero.
            //    if (candidateId >= shapeNodeIdMax) {
            //        stats.reusedMutNodes++;
            //    }
            return std::make_pair(NodeIDList{candidateId}, static_cast<NodeIDSizeT>(0));
        }
    }
    NodeIDList newNodeList{};
    size_t individualCoalCount = 0;
    // Map from an individual to which child contained it.
    std::unordered_map<NodeIDSizeT, NodeIDSizeT> individualToChild;
    // const NodeID mutNodeId = grg->makeNode(1, true);
    NodeIDList addedNodes;
    const size_t numMutSamples = mutSamples.size();
    while (!candidates.empty() && covered.size() < numMutSamples) {
        const auto& candidate = candidates.back();
        const auto candidateId = std::get<0>(candidate);
        const NodeIDList candidateSet = collector.getSamplesForCandidate(candidateId);
        release_assert(!candidateSet.empty());
        // Different candidates may cover different subsets of the sample set that
        // we are currently trying to cover. Those sample sets MUST be non-overlapping
        // or we will introduce a diamond into the graph:
        //  m-->n1-->s0
        //  m-->n2-->s0
        // However, there is no guarantee that there does not exist nodes (n1, n2)
        // that both point to a sample (or samples) that we care about, so we have to
        // track that here. We do that by only considering candidates that have no overlap
        // with our already-covered set.
        if (!setsOverlap(covered, candidateSet)) {
            // Mark all the sample nodes as covered.
            for (const auto sampleId : candidateSet) {
                covered.emplace(sampleId);
                if (ploidy == 2) {
                    auto insertPair = individualToChild.emplace(sampleId / ploidy, candidateId);
                    // The individual already existed from a _different node_, so the two samples will coalesce
                    // at the new mutation node.
                    if (!insertPair.second && candidateId != insertPair.first->second) {
                        individualCoalCount++;
                    }
                }
            }
            // Again, this check is unnecessary now
            // if (candidateId >= shapeNodeIdMax) {
            //     stats.reusedMutNodes++;
            // }

            // Use this candidate (or the nodes below it) to cover the sample subset.
            stats.reusedNodes++;
            newNodeList.push_back(candidateId);
            // grg->connect(mutNodeId, candidateId);
            stats.reusedNodeCoverage += candidateSet.size();
            if (candidateSet.size() >= stats.reuseSizeHist.size()) {
                stats.reuseSizeBiggerThanHistMax++;
            } else {
                stats.reuseSizeHist[candidateSet.size()]++;
            }
        }
        candidates.pop_back();
    }

    // Any leftovers, we just connect directly from the new mutation node to the
    // samples.
    NodeIDSet uncovered;
    for (const NodeID sampleNodeId : mutSamples) {
        const auto coveredIt = covered.find(sampleNodeId);
        if (coveredIt == covered.end()) {
            uncovered.emplace(sampleNodeId);
            // The individual had already been seen and >=1 of the samples was previously uncovered,
            // then the new node we create is going to be the coalescence location for that individual.
            if (ploidy == 2) {
                auto insertPair = individualToChild.emplace(sampleNodeId / ploidy, grg->numNodes() + batchCount);
                if (!insertPair.second) {
                    individualCoalCount++;
                }
            }
        }
    }
    NodeIDSizeT numCoals = 0;
    if (ploidy == 2) {
        numCoals = individualCoalCount;
    }

    if (!uncovered.empty()) {
        stats.numWithSingletons++;
    }

    stats.maxSingletons = std::max(uncovered.size(), stats.maxSingletons);

    for (auto sampleNodeId : uncovered) {
        newNodeList.push_back(sampleNodeId);
        // grg->connect(mutNodeId, sampleNodeId);
        stats.singletonSampleEdges++;
    }
    // This node needs to be last, for the way we update things.
    return std::make_pair(std::move(newNodeList), numCoals);
}

/// Serially apply all batched mutation mapping modifications to the grg
static NodeIDList applyBatchModifications(const MutableGRGPtr& grg,
                                          const std::vector<std::pair<NodeIDList, NodeIDSizeT>>& batchResults,
                                          const std::vector<Mutation>& mutations) {
    NodeIDList added;
    size_t numMutations = mutations.size();
    for (size_t i = 0; i < numMutations; ++i) {
        const auto& nodes = batchResults[i].first;
        NodeIDSizeT coalCount = batchResults[i].second;
        const Mutation& mut = mutations[i];
        if (nodes.size() == 1) {
            grg->addMutation(mut, nodes[0]);
        } else {
            NodeID nid = grg->makeNode(1, true);
            grg->addMutation(mut, nid);
            added.push_back(nid);
            for (auto ptr : nodes) {
                grg->connect(nid, ptr);
            }
            grg->setNumIndividualCoals(nid, coalCount);
        }
    }
    return added;
}


/**
 * Top-level entry point for mapping mutations to a GRG in parallel.
 *
 * Divides mutations into batches, processes them in parallel using
 * process_batch_par(), spawning num_threads threads which each compute
 * graph modifications for batchSize mutations. It then merges
 * all of these modifications into the graph serially.
 *
 */
MutationMappingStats mapMutations(const MutableGRGPtr& grg, MutationIterator& mutations, const size_t numThreads) {
    auto operationStartTime = std::chrono::high_resolution_clock::now();
#define START_TIMING_OPERATION() operationStartTime = std::chrono::high_resolution_clock::now();
#define EMIT_TIMING_MESSAGE(msg)                                                                                       \
    do {                                                                                                               \
        std::cout << msg                                                                                               \
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - \
                                                                           operationStartTime)                         \
                         .count()                                                                                      \
                  << " ms" << std::endl;                                                                               \
    } while (0)

    if (!grg->nodesAreOrdered()) {
        throw ApiMisuseFailure(
            "mapMutations can only be used on GRGs with ordered nodes; saving/loading a GRG will do this.");
    }

    omp_set_num_threads(numThreads);

    MutationMappingStats stats;
    stats.reuseSizeHist.resize(STATS_HIST_SIZE, 0);
    stats.totalMutations = mutations.countMutations();

    // For the whole graph, count the number of samples under each node.
    DfsSampleCountVisitor countVisitor;
    fastCompleteDFS(grg, countVisitor);
    std::vector<NodeIDSizeT>& sampleCounts = countVisitor.m_sampleCounts;

    std::cout << "Mapping " << stats.totalMutations << " mutations\n";
    const size_t onePercent = (stats.totalMutations / ONE_HUNDRED_PERCENT) + 1;

    // Note: this is made redundant by the batch application
    // The low-water mark for nodes. If a NodeID is greater than or equal to this, then it
    // is a newly added (mutation) node. 
    const NodeID shapeNodeIdMax = grg->numNodes();

    // For each mutation, perform a topological bottom-up traversal from the sample
    // nodes of interest, and collect all nodes that reach a subset of those nodes.

    size_t _ignored = 0;
    MutationAndSamples unmapped;

    std::vector<std::pair<NodeIDList, NodeIDSizeT>> batchTasks(stats.totalMutations);

    std::vector<MutationMappingStats> localStats(stats.totalMutations);
    std::vector<Mutation> mutationList{};
    mutationList.reserve(stats.totalMutations);
    std::vector<Mutation> invalidMutations{};
    size_t taskNum = 0;
    std::atomic<int> completed(0);

#pragma omp parallel
    {
#pragma omp single nowait
        {
            size_t prevPrintPercent = 0;
            size_t lastSamplesetSize = 0;
            while (mutations.next(unmapped, _ignored)) {
                if (unmapped.samples.empty()) {
                    // TODO: fix so there's no data races
                    lastSamplesetSize = unmapped.samples.size();
                    stats.emptyMutations++;
                    invalidMutations.push_back(unmapped.mutation);
                } else {
                    mutationList.push_back(unmapped.mutation);
                    stats.samplesProcessed += unmapped.samples.size();
                    if (unmapped.samples.size() == 1) {
                        stats.mutationsWithOneSample++;
                    }

                    NodeIDList sample = std::move(unmapped.samples);
#pragma omp task firstprivate(taskNum, sample)
                    {
                        localStats[taskNum].reuseSizeHist.resize(STATS_HIST_SIZE);
                        batchTasks[taskNum] = greedyAddMutationImmutable(
                            grg, sampleCounts, sample, localStats[taskNum], taskNum);
                        completed.fetch_add(1, std::memory_order_relaxed);
                    }
                    taskNum++;

                    size_t numCompleted = completed.load() + invalidMutations.size();
                    if (numCompleted != 0) {
                        size_t currPercent = numCompleted % onePercent;
                        if (currPercent > prevPrintPercent) {
                            for (size_t p = prevPrintPercent + 1; p <= currPercent; p++) {
                                std::cout << p << "% done\n";
                                if ((p % (EMIT_STATS_AT_PERCENT * onePercent)) == 0) {
                                    std::cout << "Last mutation sampleset size: " << lastSamplesetSize << "\n";
                                    std::cout << "GRG nodes: " << grg->numNodes() << "\n";
                                    std::cout << "GRG edges: " << grg->numEdges() << "\n";
                                    stats.print(std::cout);
                                }
                                prevPrintPercent = currPercent;
                            }
                        }
                    }
                }
            }
        }

#pragma omp taskwait
    }

    batchTasks.resize(taskNum);
    localStats.resize(taskNum);

    for (auto& mutation : invalidMutations) {
        grg->addMutation(mutation, INVALID_NODE_ID);
    }

    for (auto const& threadStat : localStats) {
        stats.reusedNodes += threadStat.reusedNodes;
        stats.reusedNodeCoverage += threadStat.reusedNodeCoverage;
        stats.reusedExactly += threadStat.reusedExactly;
        stats.reusedMutNodes += threadStat.reusedMutNodes;
        stats.singletonSampleEdges += threadStat.singletonSampleEdges;
        stats.numWithSingletons += threadStat.numWithSingletons;
        stats.maxSingletons = std::max(stats.maxSingletons, threadStat.maxSingletons);
        for (size_t k = 0; k < threadStat.reuseSizeHist.size(); k++) {
            stats.reuseSizeHist[k] += threadStat.reuseSizeHist[k];
        }
    }
    NodeIDList added = applyBatchModifications(grg, batchTasks, mutationList);

    START_TIMING_OPERATION();
    grg->compact();
    EMIT_TIMING_MESSAGE("Compacting GRG edges took ");

    std::vector<NodeID> newNodes;
    // add relevant nodes to samplecounts
    for (auto nodeId : added) {
        if (nodeId >= shapeNodeIdMax) {
            newNodes.push_back(nodeId);
        }
    }
    
    
    if (!newNodes.empty()) {
        size_t oldSize = sampleCounts.size();
        sampleCounts.resize(oldSize + newNodes.size());
        for (auto nodeId : newNodes) {
            NodeIDSizeT sumSamples = 0;
            for (auto child : grg->getDownEdges(nodeId)) {
                sumSamples += sampleCounts[child];
            }
            sampleCounts[nodeId] = sumSamples;
        }
    }
    return stats;

}
}; // namespace grgl
