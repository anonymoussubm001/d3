from torchrec.distributed.test_utils.test_sharding import (
    SharderType,
)
from torchrec.distributed.types import (
    ShardingType,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel

class DistributedConfig:
    def __init__(self, sharder_type, sharding_type, kernel_type, world_size, backend, quantization):
        self.sharder_type = sharder_type
        self.sharding_type = sharding_type
        self.kernel_type = kernel_type
        self.world_size = world_size
        self.backend = backend
        self.quantization = quantization


distributed_settings = [
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 1, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 2, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 3, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 4, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 8, "nccl", False),

    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 1, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 2, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 3, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 4, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.ROW_WISE.value, EmbeddingComputeKernel.DENSE.value, 8, "nccl", False),

    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 1, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 2, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 3, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 4, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 8, "nccl", False),

    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 1, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 2, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 3, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 4, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 8, "nccl", False),

    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 1, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 2, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 3, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 4, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 8, "nccl", False),

    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 1, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 2, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 3, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 4, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 8, "nccl", False),

    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 1, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 2, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 3, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 4, "nccl", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 8, "nccl", False),

    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 1, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 2, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 3, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 4, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.DENSE.value, 8, "gloo", False),

    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 1, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 2, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 3, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 4, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.COLUMN_WISE.value, EmbeddingComputeKernel.DENSE.value, 8, "gloo", False),

    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 1, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 2, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 3, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 4, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG_COLLECTION.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 8, "gloo", False),

    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 1, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 2, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 3, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 4, "gloo", False),
    DistributedConfig(SharderType.EMBEDDING_BAG.value, ShardingType.DATA_PARALLEL.value, EmbeddingComputeKernel.DENSE.value, 8, "gloo", False),
    
    DistributedConfig("quant_embedding_bag", ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.QUANT.value, 1, "gloo", True),
    DistributedConfig("quant_embedding_bag", ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.QUANT.value, 2, "gloo", True),
    DistributedConfig("quant_embedding_bag", ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.QUANT.value, 3, "gloo", True),
    DistributedConfig("quant_embedding_bag", ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.QUANT.value, 4, "gloo", True),

    DistributedConfig("quant_embedding_bag", ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.QUANT.value, 1, "nccl", True),
    DistributedConfig("quant_embedding_bag", ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.QUANT.value, 2, "nccl", True),
    DistributedConfig("quant_embedding_bag", ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.QUANT.value, 3, "nccl", True),
    DistributedConfig("quant_embedding_bag", ShardingType.TABLE_WISE.value, EmbeddingComputeKernel.QUANT.value, 4, "nccl", True),
]