/// Unified error type for Pgramma
#[derive(Debug, thiserror::Error)]
pub enum PgrammaError {
    #[error("database: {0}")]
    Db(#[from] redb::DatabaseError),

    #[error("table: {0}")]
    Table(#[from] redb::TableError),

    #[error("transaction: {0}")]
    Transaction(#[from] redb::TransactionError),

    #[error("commit: {0}")]
    Commit(#[from] redb::CommitError),

    #[error("storage: {0}")]
    Storage(#[from] redb::StorageError),

    #[error("config key not found: {0}")]
    ConfigNotFound(String),

    #[error("serialization: {0}")]
    Serialization(String),

    #[error("{0}")]
    InvalidData(String),

    #[error("embedding: {0}")]
    Embedding(String),
}

pub type Result<T> = std::result::Result<T, PgrammaError>;
