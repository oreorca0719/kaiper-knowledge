# app/auth

## OVERVIEW
DynamoDB-backed authentication module.

## STRUCTURE
```
auth/
├── __init__.py       # Empty marker
├── routes.py        # Login/logout/register endpoints
├── security.py      # Password hashing (bcrypt)
├── deps.py          # FastAPI dependencies (get_current_user, require_approved_user)
└── dynamo.py        # DynamoDB user operations
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Add auth endpoint | `routes.py` | FastAPI router |
| Modify password policy | `security.py` | bcrypt config |
| Change user model | `dynamo.py` | DynamoDB table operations |
| Add auth guard | `deps.py` | Dependency injection |

## CONVENTIONS
- Uses Starlette SessionMiddleware
- Passwords hashed with bcrypt
- DynamoDB for user storage (requires AWS credentials)

## ANTI-PATTERNS
- **DO NOT** hardcode credentials in code
- **DO NOT** expose DynamoDB errors to clients
