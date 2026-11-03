from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlmodel import Session, select

from core.db import User, get_session
from core.security import create_token, current_user, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


class SignUp(BaseModel):
    email: EmailStr
    password: str
    name: str = ""
    role: str = "user"


class Login(BaseModel):
    email: EmailStr
    password: str


@router.post("/signup")
def signup(body: SignUp, session: Session = Depends(get_session)):
    if session.exec(select(User).where(User.email == body.email)).first():
        raise HTTPException(400, "Email already registered")
    user = User(email=body.email, name=body.name or body.email.split("@")[0], role=body.role,
                password_hash=hash_password(body.password))
    session.add(user); session.commit(); session.refresh(user)
    return {"token": create_token(user.email, role=user.role), "user": {"id": user.id, "email": user.email, "name": user.name, "role": user.role}}


@router.post("/login")
def login(body: Login, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == body.email)).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    return {"token": create_token(user.email, role=user.role), "user": {"id": user.id, "email": user.email, "name": user.name, "role": user.role}}


@router.get("/me")
def me(user: User = Depends(current_user)):
    return {"id": user.id, "email": user.email, "name": user.name, "role": user.role}
