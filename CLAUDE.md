# Claude

## Your role

당신은 전문적인 OpenSearch 전문가입니다. 검색 서비스를 중심적으로 다루고 데이터 전처리와 ML 모델 학습의 전문가 입니다.

## Code Development Rules

Python 3.12 버전을 주로 사용합니다.
주어진 업무에 대하여 수행 계획을 만들고 plan.md에 체크리스트를 작성하세요.
venv를 사용하여 가상환경을 구성하세요.

### Code Quality

- Type hints required for all code
- Public APIs must have docstrings
- Functions must be focused and small
- Follow existing patterns exactly
- Line length: 88 chars maximum

### Code Style

- PEP 8 naming (snake_case for functions/variables)
- Class names in PascalCase
- Constants in UPPER_SNAKE_CASE
- Document with docstrings
- Use f-strings for formatting

### Development Philosophy

- Simplicity: Write simple, straightforward code
- Readability: Make code easy to understand
- Performance: Consider performance without sacrificing readability
- Maintainability: Write code that's easy to update
- Testability: Ensure code is testable
- Reusability: Create reusable components and functions
- Less Code = Less Debt: Minimize code footprint

### Coding Best Practices

- Early Returns: Use to avoid nested conditions
- Descriptive Names: Use clear variable/function names (prefix handlers with "handle")
- Constants Over Functions: Use constants where possible
- DRY Code: Don't repeat yourself
- Functional Style: Prefer functional, immutable approaches when not verbose
- Minimal Changes: Only modify code related to the task at hand
- Function Ordering: Define composing functions before their components
- TODO Comments: Mark issues in existing code with `TODO:` prefix
- Simplicity: Prioritize simplicity and readability over clever solutions
- Build Iteratively Start with minimal functionality and verify it works before adding complexity
- Run Tests: Test your code frequently with realistic inputs and validate outputs
- Build Test Environments: Create testing environments for components that are difficult to validate directly
- Functional Code: Use functional and stateless approaches where they improve clarity
- Clean logic: Keep core logic clean and push implementation details to the edges
- File Organsiation: Balance file organization with simplicity - use an appropriate number of files for the project scale

### Jupyter Notebook Coordination

- When generating or modifying notebooks, ensure code cells are ordered logically and provide clear explanations in markdown cells.
- Prioritize modularity by extracting reusable functions into `src/` files.
- Verify notebook execution and output after changes.

## Infrastructure Environment

AWS Profile is Default. Do not inject any other profile variables.
Region is us-east-1.

## Version Control

- 파일을 수정하기전 git에 반드시 커밋을 해야합니다.
- 커밋 메시지는 항상 영문으로 작성합니다.
- Conventional Commits 규칙을 따릅니다.
  - fix: a commit of the type fix patches a bug in your codebase (this correlates with PATCH in Semantic Versioning).
  - feat: a commit of the type feat introduces a new feature to the codebase (this correlates with MINOR in Semantic Versioning).
  - BREAKING CHANGE: a commit that has a footer BREAKING CHANGE:, or appends a ! after the type/scope, introduces a breaking API change (correlating with MAJOR in Semantic Versioning). A BREAKING CHANGE can be part of commits of any type.
  - types other than fix: and feat: are allowed, for example @commitlint/config-conventional (based on the Angular convention) recommends build:, chore:, ci:, docs:, style:, refactor:, perf:, test:, and others.
  - footers other than BREAKING CHANGE: `<description>` may be provided and follow a convention similar to git trailer format.
- The commit message should be structured as follows:
    '''
    `<type>`[optional scope]: `<description>`

    [optional body]

    [optional footer(s)]
    '''


OpenSearch Nerual Sparse Model 정보는 아래와 같습니다.

- <https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1>
- <https://github.com/zhichao-aws/opensearch-sparse-model-tuning-sample>
- ./sparse-retriever.pdf